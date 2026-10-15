import Stage_online_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_online_0

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
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix modulus_0 8 54 25
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix modulus_0 8 1 50
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix modulus_1 8 1 25
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix modulus_1 8 1 1
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix modulus_0 8 1 625
local notation "mxxMat17" => Mxx.Primitives.ExactMatrix modulus_0 8 25 625
local notation "mxxMat18" => Mxx.Primitives.ExactMatrix modulus_0 8 625 625
local notation "mxxMat19" => Mxx.Primitives.ExactMatrix modulus_0 8 25 1
local notation "mxxMat20" => Mxx.Primitives.ExactMatrix modulus_2 8 1 25
local notation "mxxMat21" => Mxx.Primitives.ExactMatrix modulus_2 8 1 1
local notation "mxxMat22" => Mxx.Primitives.ExactMatrix modulus_3 8 1 25
local notation "mxxMat23" => Mxx.Primitives.ExactMatrix modulus_3 8 1 1
local notation "mxxMat24" => Mxx.Primitives.ExactMatrix modulus_4 8 1 25
local notation "mxxMat25" => Mxx.Primitives.ExactMatrix modulus_4 8 1 1
local notation "mxxMat26" => Mxx.Primitives.ExactMatrix modulus_5 8 1 25
local notation "mxxMat27" => Mxx.Primitives.ExactMatrix modulus_5 8 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_30.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit
  witness : parallel_generatedRoot_30.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_30Ctx" => (parallel_generatedRoot_30.Ctx backend hashModel params i_0)

def parallel_generatedRoot_30.w_0_0  (ctx : parallel_generatedRoot_30Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_30.w_1_0  (ctx : parallel_generatedRoot_30Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_30.w_2_0  (ctx : parallel_generatedRoot_30Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_30.w_3_0  (ctx : parallel_generatedRoot_30Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_30.w_4_0  (ctx : parallel_generatedRoot_30Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_30.w_5_0  (ctx : parallel_generatedRoot_30Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_30.w_6_0  (ctx : parallel_generatedRoot_30Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_30.w_6_1  (ctx : parallel_generatedRoot_30Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_30.w_6_2  (ctx : parallel_generatedRoot_30Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_30.w_6_3  (ctx : parallel_generatedRoot_30Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_30.w_6_4  (ctx : parallel_generatedRoot_30Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_30.w_6_5  (ctx : parallel_generatedRoot_30Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_30.w_7_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_30.w_8_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_30.w_9_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_30.w_7_0 ctx) (parallel_generatedRoot_30.w_8_0 ctx)

def parallel_generatedRoot_30.w_10_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_30.w_11_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_30.w_12_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_30.w_10_0 ctx) (parallel_generatedRoot_30.w_11_0 ctx)

def parallel_generatedRoot_30.w_13_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_30.w_14_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_30.w_15_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_30.w_13_0 ctx) (parallel_generatedRoot_30.w_14_0 ctx)

def parallel_generatedRoot_30.w_16_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_30.w_17_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_30.w_18_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_30.w_16_0 ctx) (parallel_generatedRoot_30.w_17_0 ctx)

def parallel_generatedRoot_30.w_19_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_30.w_20_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_30.w_21_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_30.w_19_0 ctx) (parallel_generatedRoot_30.w_20_0 ctx)

def parallel_generatedRoot_30.w_22_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_30.w_23_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_30.w_24_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_30.w_22_0 ctx) (parallel_generatedRoot_30.w_23_0 ctx)

def parallel_generatedRoot_30.w_25_0  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_30.w_25_1  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_30.w_25_2  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_30.w_25_3  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_30.w_25_4  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_30.w_25_5  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_30.outs  (ctx : parallel_generatedRoot_30Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_30.w_25_0 ctx), ((parallel_generatedRoot_30.w_25_1 ctx), ((parallel_generatedRoot_30.w_25_2 ctx), ((parallel_generatedRoot_30.w_25_3 ctx), ((parallel_generatedRoot_30.w_25_4 ctx), ((parallel_generatedRoot_30.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_30.Facts_0  (ctx : parallel_generatedRoot_30Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_30_6 backend hashModel params i_0 i ((parallel_generatedRoot_30.w_0_0 ctx), ((parallel_generatedRoot_30.w_1_0 ctx), ((parallel_generatedRoot_30.w_2_0 ctx), ((parallel_generatedRoot_30.w_3_0 ctx), ((parallel_generatedRoot_30.w_4_0 ctx), ((parallel_generatedRoot_30.w_5_0 ctx), ())))))) (((parallel_generatedRoot_30.w_6_0 ctx) i), (((parallel_generatedRoot_30.w_6_1 ctx) i), (((parallel_generatedRoot_30.w_6_2 ctx) i), (((parallel_generatedRoot_30.w_6_3 ctx) i), (((parallel_generatedRoot_30.w_6_4 ctx) i), (((parallel_generatedRoot_30.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_0 ctx) (0) (parallel_generatedRoot_30.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_0 ctx) (1) (parallel_generatedRoot_30.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_1 ctx) (0) (parallel_generatedRoot_30.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_1 ctx) (1) (parallel_generatedRoot_30.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_2 ctx) (0) (parallel_generatedRoot_30.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_2 ctx) (1) (parallel_generatedRoot_30.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_3 ctx) (0) (parallel_generatedRoot_30.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_3 ctx) (1) (parallel_generatedRoot_30.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_4 ctx) (0) (parallel_generatedRoot_30.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_4 ctx) (1) (parallel_generatedRoot_30.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_5 ctx) (0) (parallel_generatedRoot_30.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_30.w_6_5 ctx) (1) (parallel_generatedRoot_30.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_30_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_30.w_6_0 ctx), ((parallel_generatedRoot_30.w_6_1 ctx), ((parallel_generatedRoot_30.w_6_2 ctx), ((parallel_generatedRoot_30.w_6_3 ctx), ((parallel_generatedRoot_30.w_6_4 ctx), ((parallel_generatedRoot_30.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_30.w_9_0 ctx), ((parallel_generatedRoot_30.w_12_0 ctx), ((parallel_generatedRoot_30.w_15_0 ctx), ((parallel_generatedRoot_30.w_18_0 ctx), ((parallel_generatedRoot_30.w_21_0 ctx), ((parallel_generatedRoot_30.w_24_0 ctx), ())))))) ((parallel_generatedRoot_30.w_25_0 ctx), ((parallel_generatedRoot_30.w_25_1 ctx), ((parallel_generatedRoot_30.w_25_2 ctx), ((parallel_generatedRoot_30.w_25_3 ctx), ((parallel_generatedRoot_30.w_25_4 ctx), ((parallel_generatedRoot_30.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_30.outs ctx

theorem parallel_generatedRoot_30.facts_0  {ctx : parallel_generatedRoot_30Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_30.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_30.w_0_0 ctx) (parallel_generatedRoot_30.w_1_0 ctx) (parallel_generatedRoot_30.w_2_0 ctx) (parallel_generatedRoot_30.w_3_0 ctx) (parallel_generatedRoot_30.w_4_0 ctx) (parallel_generatedRoot_30.w_5_0 ctx) (parallel_generatedRoot_30.w_6_0 ctx) (parallel_generatedRoot_30.w_6_1 ctx) (parallel_generatedRoot_30.w_6_2 ctx) (parallel_generatedRoot_30.w_6_3 ctx) (parallel_generatedRoot_30.w_6_4 ctx) (parallel_generatedRoot_30.w_6_5 ctx) (parallel_generatedRoot_30.w_7_0 ctx) (parallel_generatedRoot_30.w_8_0 ctx) (parallel_generatedRoot_30.w_9_0 ctx) (parallel_generatedRoot_30.w_10_0 ctx) (parallel_generatedRoot_30.w_11_0 ctx) (parallel_generatedRoot_30.w_12_0 ctx) (parallel_generatedRoot_30.w_13_0 ctx) (parallel_generatedRoot_30.w_14_0 ctx) (parallel_generatedRoot_30.w_15_0 ctx) (parallel_generatedRoot_30.w_16_0 ctx) (parallel_generatedRoot_30.w_17_0 ctx) (parallel_generatedRoot_30.w_18_0 ctx) (parallel_generatedRoot_30.w_19_0 ctx) (parallel_generatedRoot_30.w_20_0 ctx) (parallel_generatedRoot_30.w_21_0 ctx) (parallel_generatedRoot_30.w_22_0 ctx) (parallel_generatedRoot_30.w_23_0 ctx) (parallel_generatedRoot_30.w_24_0 ctx) (parallel_generatedRoot_30.w_25_0 ctx) (parallel_generatedRoot_30.w_25_1 ctx) (parallel_generatedRoot_30.w_25_2 ctx) (parallel_generatedRoot_30.w_25_3 ctx) (parallel_generatedRoot_30.w_25_4 ctx) (parallel_generatedRoot_30.w_25_5 ctx) outputs) :
    parallel_generatedRoot_30.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_30.Facts  (ctx : parallel_generatedRoot_30Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_30.Facts_0 ctx outputs

theorem parallel_generatedRoot_30.facts  (ctx : parallel_generatedRoot_30Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_30.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_30.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_30.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_34.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit
  witness : parallel_generatedRoot_34.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_34Ctx" => (parallel_generatedRoot_34.Ctx backend hashModel params i_0)

def parallel_generatedRoot_34.w_0_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_34.w_1_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_34.w_2_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_34.w_3_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_34.w_4_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_34.w_5_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_34.w_6_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_34.w_6_1  (ctx : parallel_generatedRoot_34Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_34.w_6_2  (ctx : parallel_generatedRoot_34Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_34.w_6_3  (ctx : parallel_generatedRoot_34Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_34.w_6_4  (ctx : parallel_generatedRoot_34Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_34.w_6_5  (ctx : parallel_generatedRoot_34Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_34.w_7_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_34.w_8_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_34.w_9_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_7_0 ctx) (parallel_generatedRoot_34.w_8_0 ctx)

def parallel_generatedRoot_34.w_10_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_34.w_11_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_34.w_12_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_10_0 ctx) (parallel_generatedRoot_34.w_11_0 ctx)

def parallel_generatedRoot_34.w_13_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_34.w_14_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_34.w_15_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_13_0 ctx) (parallel_generatedRoot_34.w_14_0 ctx)

def parallel_generatedRoot_34.w_16_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_34.w_17_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_34.w_18_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_16_0 ctx) (parallel_generatedRoot_34.w_17_0 ctx)

def parallel_generatedRoot_34.w_19_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_34.w_20_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_34.w_21_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_19_0 ctx) (parallel_generatedRoot_34.w_20_0 ctx)

def parallel_generatedRoot_34.w_22_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_34.w_23_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_34.w_24_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_22_0 ctx) (parallel_generatedRoot_34.w_23_0 ctx)

def parallel_generatedRoot_34.w_25_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_34.w_25_1  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_34.w_25_2  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_34.w_25_3  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_34.w_25_4  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_34.w_25_5  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_34.outs  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_34.w_25_0 ctx), ((parallel_generatedRoot_34.w_25_1 ctx), ((parallel_generatedRoot_34.w_25_2 ctx), ((parallel_generatedRoot_34.w_25_3 ctx), ((parallel_generatedRoot_34.w_25_4 ctx), ((parallel_generatedRoot_34.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_34.Facts_0  (ctx : parallel_generatedRoot_34Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_34_6 backend hashModel params i_0 i ((parallel_generatedRoot_34.w_0_0 ctx), ((parallel_generatedRoot_34.w_1_0 ctx), ((parallel_generatedRoot_34.w_2_0 ctx), ((parallel_generatedRoot_34.w_3_0 ctx), ((parallel_generatedRoot_34.w_4_0 ctx), ((parallel_generatedRoot_34.w_5_0 ctx), ())))))) (((parallel_generatedRoot_34.w_6_0 ctx) i), (((parallel_generatedRoot_34.w_6_1 ctx) i), (((parallel_generatedRoot_34.w_6_2 ctx) i), (((parallel_generatedRoot_34.w_6_3 ctx) i), (((parallel_generatedRoot_34.w_6_4 ctx) i), (((parallel_generatedRoot_34.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_0 ctx) (0) (parallel_generatedRoot_34.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_0 ctx) (1) (parallel_generatedRoot_34.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_1 ctx) (0) (parallel_generatedRoot_34.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_1 ctx) (1) (parallel_generatedRoot_34.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_2 ctx) (0) (parallel_generatedRoot_34.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_2 ctx) (1) (parallel_generatedRoot_34.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_3 ctx) (0) (parallel_generatedRoot_34.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_3 ctx) (1) (parallel_generatedRoot_34.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_4 ctx) (0) (parallel_generatedRoot_34.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_4 ctx) (1) (parallel_generatedRoot_34.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_5 ctx) (0) (parallel_generatedRoot_34.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_6_5 ctx) (1) (parallel_generatedRoot_34.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_34_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_34.w_6_0 ctx), ((parallel_generatedRoot_34.w_6_1 ctx), ((parallel_generatedRoot_34.w_6_2 ctx), ((parallel_generatedRoot_34.w_6_3 ctx), ((parallel_generatedRoot_34.w_6_4 ctx), ((parallel_generatedRoot_34.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_34.w_9_0 ctx), ((parallel_generatedRoot_34.w_12_0 ctx), ((parallel_generatedRoot_34.w_15_0 ctx), ((parallel_generatedRoot_34.w_18_0 ctx), ((parallel_generatedRoot_34.w_21_0 ctx), ((parallel_generatedRoot_34.w_24_0 ctx), ())))))) ((parallel_generatedRoot_34.w_25_0 ctx), ((parallel_generatedRoot_34.w_25_1 ctx), ((parallel_generatedRoot_34.w_25_2 ctx), ((parallel_generatedRoot_34.w_25_3 ctx), ((parallel_generatedRoot_34.w_25_4 ctx), ((parallel_generatedRoot_34.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_34.outs ctx

theorem parallel_generatedRoot_34.facts_0  {ctx : parallel_generatedRoot_34Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_34.w_0_0 ctx) (parallel_generatedRoot_34.w_1_0 ctx) (parallel_generatedRoot_34.w_2_0 ctx) (parallel_generatedRoot_34.w_3_0 ctx) (parallel_generatedRoot_34.w_4_0 ctx) (parallel_generatedRoot_34.w_5_0 ctx) (parallel_generatedRoot_34.w_6_0 ctx) (parallel_generatedRoot_34.w_6_1 ctx) (parallel_generatedRoot_34.w_6_2 ctx) (parallel_generatedRoot_34.w_6_3 ctx) (parallel_generatedRoot_34.w_6_4 ctx) (parallel_generatedRoot_34.w_6_5 ctx) (parallel_generatedRoot_34.w_7_0 ctx) (parallel_generatedRoot_34.w_8_0 ctx) (parallel_generatedRoot_34.w_9_0 ctx) (parallel_generatedRoot_34.w_10_0 ctx) (parallel_generatedRoot_34.w_11_0 ctx) (parallel_generatedRoot_34.w_12_0 ctx) (parallel_generatedRoot_34.w_13_0 ctx) (parallel_generatedRoot_34.w_14_0 ctx) (parallel_generatedRoot_34.w_15_0 ctx) (parallel_generatedRoot_34.w_16_0 ctx) (parallel_generatedRoot_34.w_17_0 ctx) (parallel_generatedRoot_34.w_18_0 ctx) (parallel_generatedRoot_34.w_19_0 ctx) (parallel_generatedRoot_34.w_20_0 ctx) (parallel_generatedRoot_34.w_21_0 ctx) (parallel_generatedRoot_34.w_22_0 ctx) (parallel_generatedRoot_34.w_23_0 ctx) (parallel_generatedRoot_34.w_24_0 ctx) (parallel_generatedRoot_34.w_25_0 ctx) (parallel_generatedRoot_34.w_25_1 ctx) (parallel_generatedRoot_34.w_25_2 ctx) (parallel_generatedRoot_34.w_25_3 ctx) (parallel_generatedRoot_34.w_25_4 ctx) (parallel_generatedRoot_34.w_25_5 ctx) outputs) :
    parallel_generatedRoot_34.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_34.Facts  (ctx : parallel_generatedRoot_34Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_34.Facts_0 ctx outputs

theorem parallel_generatedRoot_34.facts  (ctx : parallel_generatedRoot_34Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_34.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_34.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_35.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_35.InputType)
  witness : parallel_generatedRoot_35.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_35Ctx" => (parallel_generatedRoot_35.Ctx backend hashModel params i_0)

def parallel_generatedRoot_35.w_1_0  (ctx : parallel_generatedRoot_35Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_35.w_3_0  (ctx : parallel_generatedRoot_35Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_35.w_5_0  (ctx : parallel_generatedRoot_35Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_35.w_7_0  (ctx : parallel_generatedRoot_35Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_35.w_9_0  (ctx : parallel_generatedRoot_35Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_35.w_10_0  (ctx : parallel_generatedRoot_35Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_35.w_11_0  (ctx : parallel_generatedRoot_35Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_35.w_12_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_35.w_13_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_35.w_14_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_35.w_15_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_35.w_16_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_35.w_17_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_35.w_0_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_35.w_2_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_35.w_4_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_35.w_6_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_35.w_8_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_35.w_18_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_35.w_18_1  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_35.w_18_2  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_35.w_18_3  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_35.w_18_4  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_35.w_18_5  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_35.w_18_6  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_35.w_18_7  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_35.w_18_8  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_35.w_18_9  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_35.w_19_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_35.w_20_0  (ctx : parallel_generatedRoot_35Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_35.w_21_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_35.w_22_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_35.w_23_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_35.w_23_concat_1  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_35.w_24_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_35.w_25_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_35.w_26_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_35.w_27_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_35.w_27_concat_1  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_35.w_28_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_35.w_29_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_35.w_30_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_35.w_31_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_35.w_32_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_35.w_32_concat_1  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_35.w_33_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_35.w_34_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_35.w_35_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_35.w_36_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_35.w_36_concat_1  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_35.w_37_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_35.w_38_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_35.w_39_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_35.w_40_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_35.w_41_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_35.w_37_0 ctx) (parallel_generatedRoot_35.w_40_0 ctx)

def parallel_generatedRoot_35.w_42_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_35.w_43_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_35.w_44_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_35.w_45_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_35.w_46_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_35.w_47_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_35.w_28_0 ctx) (parallel_generatedRoot_35.w_46_0 ctx)

def parallel_generatedRoot_35.w_48_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_35.w_49_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_18_2 ctx) (parallel_generatedRoot_35.w_48_0 ctx)

def parallel_generatedRoot_35.w_50_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_35.w_51_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_35.w_52_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_35.w_53_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_35.w_37_0 ctx) (parallel_generatedRoot_35.w_52_0 ctx)

def parallel_generatedRoot_35.w_54_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_35.w_55_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_35.w_56_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_35.w_57_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_35.w_58_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_35.w_59_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_35.w_28_0 ctx) (parallel_generatedRoot_35.w_58_0 ctx)

def parallel_generatedRoot_35.w_60_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_35.w_61_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_18_3 ctx) (parallel_generatedRoot_35.w_60_0 ctx)

def parallel_generatedRoot_35.w_62_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_35.w_63_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_18_4 ctx) (parallel_generatedRoot_35.w_62_0 ctx)

def parallel_generatedRoot_35.w_64_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_35.w_65_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_18_5 ctx) (parallel_generatedRoot_35.w_64_0 ctx)

def parallel_generatedRoot_35.w_66_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_35.w_67_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_35.w_68_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_66_0 ctx) (parallel_generatedRoot_35.w_67_0 ctx)

def parallel_generatedRoot_35.w_69_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_18_0 ctx) (parallel_generatedRoot_35.w_68_0 ctx)

def parallel_generatedRoot_35.w_70_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_35.w_71_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_35.w_72_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_70_0 ctx) (parallel_generatedRoot_35.w_71_0 ctx)

def parallel_generatedRoot_35.w_73_0  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_35.w_18_1 ctx) (parallel_generatedRoot_35.w_72_0 ctx)

def parallel_generatedRoot_35.outs  (ctx : parallel_generatedRoot_35Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_35.w_49_0 ctx), ((parallel_generatedRoot_35.w_61_0 ctx), ((parallel_generatedRoot_35.w_63_0 ctx), ((parallel_generatedRoot_35.w_65_0 ctx), ((parallel_generatedRoot_35.w_69_0 ctx), ((parallel_generatedRoot_35.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_35.Facts_0  (ctx : parallel_generatedRoot_35Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_35.w_1_0 ctx) (0) (parallel_generatedRoot_35.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_35.w_3_0 ctx) (0) (parallel_generatedRoot_35.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_35.w_5_0 ctx) (0) (parallel_generatedRoot_35.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_35.w_7_0 ctx) (0) (parallel_generatedRoot_35.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_35_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_35.w_9_0 ctx), ((parallel_generatedRoot_35.w_10_0 ctx), ((parallel_generatedRoot_35.w_11_0 ctx), ((parallel_generatedRoot_35.w_12_0 ctx), ((parallel_generatedRoot_35.w_13_0 ctx), ((parallel_generatedRoot_35.w_14_0 ctx), ((parallel_generatedRoot_35.w_15_0 ctx), ((parallel_generatedRoot_35.w_16_0 ctx), ((parallel_generatedRoot_35.w_17_0 ctx), ((parallel_generatedRoot_35.w_1_0 ctx), ((parallel_generatedRoot_35.w_3_0 ctx), ((parallel_generatedRoot_35.w_5_0 ctx), ((parallel_generatedRoot_35.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_35.w_0_0 ctx), ((parallel_generatedRoot_35.w_0_0 ctx), ((parallel_generatedRoot_35.w_0_0 ctx), ((parallel_generatedRoot_35.w_0_0 ctx), ((parallel_generatedRoot_35.w_0_0 ctx), ((parallel_generatedRoot_35.w_0_0 ctx), ((parallel_generatedRoot_35.w_2_0 ctx), ((parallel_generatedRoot_35.w_4_0 ctx), ((parallel_generatedRoot_35.w_6_0 ctx), ((parallel_generatedRoot_35.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_35.w_18_0 ctx), ((parallel_generatedRoot_35.w_18_1 ctx), ((parallel_generatedRoot_35.w_18_2 ctx), ((parallel_generatedRoot_35.w_18_3 ctx), ((parallel_generatedRoot_35.w_18_4 ctx), ((parallel_generatedRoot_35.w_18_5 ctx), ((parallel_generatedRoot_35.w_18_6 ctx), ((parallel_generatedRoot_35.w_18_7 ctx), ((parallel_generatedRoot_35.w_18_8 ctx), ((parallel_generatedRoot_35.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_35.w_9_0 ctx) (7) (parallel_generatedRoot_35.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_35.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_35.w_10_0 ctx)) (parallel_generatedRoot_35.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_35.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_35.w_10_0 ctx)) (parallel_generatedRoot_35.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_19_0 ctx) (parallel_generatedRoot_35.w_21_0 ctx) (parallel_generatedRoot_35.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_23_concat_1 ctx) (parallel_generatedRoot_35.w_22_0 ctx) (parallel_generatedRoot_35.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_35.w_11_0 ctx) (7) (parallel_generatedRoot_35.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_35.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_35.w_10_0 ctx)) (parallel_generatedRoot_35.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_35.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_35.w_10_0 ctx)) (parallel_generatedRoot_35.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_24_0 ctx) (parallel_generatedRoot_35.w_25_0 ctx) (parallel_generatedRoot_35.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_27_concat_1 ctx) (parallel_generatedRoot_35.w_26_0 ctx) (parallel_generatedRoot_35.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_35.w_23_0 ctx) (parallel_generatedRoot_35.w_27_0 ctx) (parallel_generatedRoot_35.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_35.w_12_0 ctx) (parallel_generatedRoot_35.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_35.w_13_0 ctx) (parallel_generatedRoot_35.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_35.w_14_0 ctx) (parallel_generatedRoot_35.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_29_0 ctx) (parallel_generatedRoot_35.w_30_0 ctx) (parallel_generatedRoot_35.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_32_concat_1 ctx) (parallel_generatedRoot_35.w_31_0 ctx) (parallel_generatedRoot_35.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_35.facts_0  {ctx : parallel_generatedRoot_35Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_35.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_35.w_1_0 ctx) (parallel_generatedRoot_35.w_3_0 ctx) (parallel_generatedRoot_35.w_5_0 ctx) (parallel_generatedRoot_35.w_7_0 ctx) (parallel_generatedRoot_35.w_9_0 ctx) (parallel_generatedRoot_35.w_10_0 ctx) (parallel_generatedRoot_35.w_11_0 ctx) (parallel_generatedRoot_35.w_12_0 ctx) (parallel_generatedRoot_35.w_13_0 ctx) (parallel_generatedRoot_35.w_14_0 ctx) (parallel_generatedRoot_35.w_15_0 ctx) (parallel_generatedRoot_35.w_16_0 ctx) (parallel_generatedRoot_35.w_17_0 ctx) (parallel_generatedRoot_35.w_0_0 ctx) (parallel_generatedRoot_35.w_2_0 ctx) (parallel_generatedRoot_35.w_4_0 ctx) (parallel_generatedRoot_35.w_6_0 ctx) (parallel_generatedRoot_35.w_8_0 ctx) (parallel_generatedRoot_35.w_18_0 ctx) (parallel_generatedRoot_35.w_18_1 ctx) (parallel_generatedRoot_35.w_18_2 ctx) (parallel_generatedRoot_35.w_18_3 ctx) (parallel_generatedRoot_35.w_18_4 ctx) (parallel_generatedRoot_35.w_18_5 ctx) (parallel_generatedRoot_35.w_18_6 ctx) (parallel_generatedRoot_35.w_18_7 ctx) (parallel_generatedRoot_35.w_18_8 ctx) (parallel_generatedRoot_35.w_18_9 ctx) (parallel_generatedRoot_35.w_19_0 ctx) (parallel_generatedRoot_35.w_20_0 ctx) (parallel_generatedRoot_35.w_21_0 ctx) (parallel_generatedRoot_35.w_22_0 ctx) (parallel_generatedRoot_35.w_23_0 ctx) (parallel_generatedRoot_35.w_23_concat_1 ctx) (parallel_generatedRoot_35.w_24_0 ctx) (parallel_generatedRoot_35.w_25_0 ctx) (parallel_generatedRoot_35.w_26_0 ctx) (parallel_generatedRoot_35.w_27_0 ctx) (parallel_generatedRoot_35.w_27_concat_1 ctx) (parallel_generatedRoot_35.w_28_0 ctx) (parallel_generatedRoot_35.w_29_0 ctx) (parallel_generatedRoot_35.w_30_0 ctx) (parallel_generatedRoot_35.w_31_0 ctx) (parallel_generatedRoot_35.w_32_0 ctx) (parallel_generatedRoot_35.w_32_concat_1 ctx)) :
    parallel_generatedRoot_35.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_35.Facts_1  (ctx : parallel_generatedRoot_35Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_35.w_15_0 ctx) (parallel_generatedRoot_35.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_35.w_16_0 ctx) (parallel_generatedRoot_35.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_35.w_17_0 ctx) (parallel_generatedRoot_35.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_33_0 ctx) (parallel_generatedRoot_35.w_34_0 ctx) (parallel_generatedRoot_35.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_36_concat_1 ctx) (parallel_generatedRoot_35.w_35_0 ctx) (parallel_generatedRoot_35.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_35.w_32_0 ctx) (parallel_generatedRoot_35.w_36_0 ctx) (parallel_generatedRoot_35.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_18_6 ctx) (parallel_generatedRoot_35.w_18_8 ctx) (parallel_generatedRoot_35.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_35.w_38_0 ctx) (parallel_generatedRoot_35.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_35.w_39_0 ctx) (parallel_generatedRoot_35.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_35.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_35.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_42_0 ctx) (parallel_generatedRoot_35.w_43_0 ctx) (parallel_generatedRoot_35.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_35.w_44_0 ctx) (parallel_generatedRoot_35.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_35.w_45_0 ctx) (parallel_generatedRoot_35.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_35.facts_1  {ctx : parallel_generatedRoot_35Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_35.constraints_1 backend (parallel_generatedRoot_35.w_15_0 ctx) (parallel_generatedRoot_35.w_16_0 ctx) (parallel_generatedRoot_35.w_17_0 ctx) (parallel_generatedRoot_35.w_18_6 ctx) (parallel_generatedRoot_35.w_18_8 ctx) (parallel_generatedRoot_35.w_32_0 ctx) (parallel_generatedRoot_35.w_33_0 ctx) (parallel_generatedRoot_35.w_34_0 ctx) (parallel_generatedRoot_35.w_35_0 ctx) (parallel_generatedRoot_35.w_36_0 ctx) (parallel_generatedRoot_35.w_36_concat_1 ctx) (parallel_generatedRoot_35.w_37_0 ctx) (parallel_generatedRoot_35.w_38_0 ctx) (parallel_generatedRoot_35.w_39_0 ctx) (parallel_generatedRoot_35.w_40_0 ctx) (parallel_generatedRoot_35.w_41_0 ctx) (parallel_generatedRoot_35.w_42_0 ctx) (parallel_generatedRoot_35.w_43_0 ctx) (parallel_generatedRoot_35.w_44_0 ctx) (parallel_generatedRoot_35.w_45_0 ctx) (parallel_generatedRoot_35.w_46_0 ctx)) :
    parallel_generatedRoot_35.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_35.Facts_2  (ctx : parallel_generatedRoot_35Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_35.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_18_7 ctx) (parallel_generatedRoot_35.w_18_9 ctx) (parallel_generatedRoot_35.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_35.w_50_0 ctx) (parallel_generatedRoot_35.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_35.w_51_0 ctx) (parallel_generatedRoot_35.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_35.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_35.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_35.w_54_0 ctx) (parallel_generatedRoot_35.w_55_0 ctx) (parallel_generatedRoot_35.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_35.w_56_0 ctx) (parallel_generatedRoot_35.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_35.w_57_0 ctx) (parallel_generatedRoot_35.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_35.w_60_0 ctx)

theorem parallel_generatedRoot_35.facts_2  {ctx : parallel_generatedRoot_35Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_35.constraints_2 backend (parallel_generatedRoot_35.w_18_7 ctx) (parallel_generatedRoot_35.w_18_9 ctx) (parallel_generatedRoot_35.w_47_0 ctx) (parallel_generatedRoot_35.w_48_0 ctx) (parallel_generatedRoot_35.w_50_0 ctx) (parallel_generatedRoot_35.w_51_0 ctx) (parallel_generatedRoot_35.w_52_0 ctx) (parallel_generatedRoot_35.w_53_0 ctx) (parallel_generatedRoot_35.w_54_0 ctx) (parallel_generatedRoot_35.w_55_0 ctx) (parallel_generatedRoot_35.w_56_0 ctx) (parallel_generatedRoot_35.w_57_0 ctx) (parallel_generatedRoot_35.w_58_0 ctx) (parallel_generatedRoot_35.w_59_0 ctx) (parallel_generatedRoot_35.w_60_0 ctx)) :
    parallel_generatedRoot_35.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_35.Facts_3  (ctx : parallel_generatedRoot_35Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_35.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_35.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_35.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_35.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_35.facts_3  {ctx : parallel_generatedRoot_35Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_35.constraints_3 (parallel_generatedRoot_35.w_41_0 ctx) (parallel_generatedRoot_35.w_47_0 ctx) (parallel_generatedRoot_35.w_59_0 ctx) (parallel_generatedRoot_35.w_62_0 ctx) (parallel_generatedRoot_35.w_64_0 ctx) (parallel_generatedRoot_35.w_66_0 ctx) (parallel_generatedRoot_35.w_67_0 ctx)) :
    parallel_generatedRoot_35.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_35.Facts_4  (ctx : parallel_generatedRoot_35Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_35.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_35.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_35.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_35.outs ctx

theorem parallel_generatedRoot_35.facts_4  {ctx : parallel_generatedRoot_35Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_35.constraints_4 (parallel_generatedRoot_35.w_49_0 ctx) (parallel_generatedRoot_35.w_53_0 ctx) (parallel_generatedRoot_35.w_59_0 ctx) (parallel_generatedRoot_35.w_61_0 ctx) (parallel_generatedRoot_35.w_63_0 ctx) (parallel_generatedRoot_35.w_65_0 ctx) (parallel_generatedRoot_35.w_69_0 ctx) (parallel_generatedRoot_35.w_70_0 ctx) (parallel_generatedRoot_35.w_71_0 ctx) (parallel_generatedRoot_35.w_73_0 ctx) outputs) :
    parallel_generatedRoot_35.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_35.Facts  (ctx : parallel_generatedRoot_35Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_35.Facts_0 ctx
  c1 : parallel_generatedRoot_35.Facts_1 ctx
  c2 : parallel_generatedRoot_35.Facts_2 ctx
  c3 : parallel_generatedRoot_35.Facts_3 ctx
  c4 : parallel_generatedRoot_35.Facts_4 ctx outputs

theorem parallel_generatedRoot_35.facts  (ctx : parallel_generatedRoot_35Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_35.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_35.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_35.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_35.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_35.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_35.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_35.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_36.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_36.InputType)
  witness : parallel_generatedRoot_36.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_36Ctx" => (parallel_generatedRoot_36.Ctx backend hashModel params i_0)

def parallel_generatedRoot_36.w_1_0  (ctx : parallel_generatedRoot_36Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_36.w_3_0  (ctx : parallel_generatedRoot_36Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_36.w_5_0  (ctx : parallel_generatedRoot_36Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_36.w_7_0  (ctx : parallel_generatedRoot_36Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_36.w_9_0  (ctx : parallel_generatedRoot_36Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_36.w_10_0  (ctx : parallel_generatedRoot_36Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_36.w_11_0  (ctx : parallel_generatedRoot_36Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_36.w_12_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_36.w_13_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_36.w_14_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_36.w_15_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_36.w_16_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_36.w_17_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_36.w_0_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_36.w_2_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_36.w_4_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_36.w_6_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_36.w_8_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_36.w_18_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_36.w_18_1  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_36.w_18_2  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_36.w_18_3  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_36.w_18_4  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_36.w_18_5  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_36.w_18_6  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_36.w_18_7  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_36.w_18_8  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_36.w_18_9  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_36.w_19_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_36.w_20_0  (ctx : parallel_generatedRoot_36Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_36.w_21_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_36.w_22_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_36.w_23_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_36.w_23_concat_1  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_36.w_24_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_36.w_25_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_36.w_26_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_36.w_27_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_36.w_27_concat_1  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_36.w_28_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_36.w_29_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_36.w_30_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_36.w_31_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_36.w_32_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_36.w_32_concat_1  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_36.w_33_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_36.w_34_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_36.w_35_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_36.w_36_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_36.w_36_concat_1  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_36.w_37_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_36.w_38_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_36.w_39_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_36.w_40_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_36.w_41_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_36.w_37_0 ctx) (parallel_generatedRoot_36.w_40_0 ctx)

def parallel_generatedRoot_36.w_42_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_36.w_43_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_36.w_44_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_36.w_45_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_36.w_46_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_36.w_47_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_36.w_28_0 ctx) (parallel_generatedRoot_36.w_46_0 ctx)

def parallel_generatedRoot_36.w_48_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_36.w_49_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_18_2 ctx) (parallel_generatedRoot_36.w_48_0 ctx)

def parallel_generatedRoot_36.w_50_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_36.w_51_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_36.w_52_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_36.w_53_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_36.w_37_0 ctx) (parallel_generatedRoot_36.w_52_0 ctx)

def parallel_generatedRoot_36.w_54_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_36.w_55_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_36.w_56_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_36.w_57_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_36.w_58_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_36.w_59_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_36.w_28_0 ctx) (parallel_generatedRoot_36.w_58_0 ctx)

def parallel_generatedRoot_36.w_60_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_36.w_61_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_18_3 ctx) (parallel_generatedRoot_36.w_60_0 ctx)

def parallel_generatedRoot_36.w_62_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_36.w_63_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_18_4 ctx) (parallel_generatedRoot_36.w_62_0 ctx)

def parallel_generatedRoot_36.w_64_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_36.w_65_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_18_5 ctx) (parallel_generatedRoot_36.w_64_0 ctx)

def parallel_generatedRoot_36.w_66_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_36.w_67_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_36.w_68_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_66_0 ctx) (parallel_generatedRoot_36.w_67_0 ctx)

def parallel_generatedRoot_36.w_69_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_18_0 ctx) (parallel_generatedRoot_36.w_68_0 ctx)

def parallel_generatedRoot_36.w_70_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_36.w_71_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_36.w_72_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_70_0 ctx) (parallel_generatedRoot_36.w_71_0 ctx)

def parallel_generatedRoot_36.w_73_0  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_36.w_18_1 ctx) (parallel_generatedRoot_36.w_72_0 ctx)

def parallel_generatedRoot_36.outs  (ctx : parallel_generatedRoot_36Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_36.w_49_0 ctx), ((parallel_generatedRoot_36.w_61_0 ctx), ((parallel_generatedRoot_36.w_63_0 ctx), ((parallel_generatedRoot_36.w_65_0 ctx), ((parallel_generatedRoot_36.w_69_0 ctx), ((parallel_generatedRoot_36.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_36.Facts_0  (ctx : parallel_generatedRoot_36Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_36.w_1_0 ctx) (0) (parallel_generatedRoot_36.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_36.w_3_0 ctx) (0) (parallel_generatedRoot_36.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_36.w_5_0 ctx) (0) (parallel_generatedRoot_36.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_36.w_7_0 ctx) (0) (parallel_generatedRoot_36.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_36_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_36.w_9_0 ctx), ((parallel_generatedRoot_36.w_10_0 ctx), ((parallel_generatedRoot_36.w_11_0 ctx), ((parallel_generatedRoot_36.w_12_0 ctx), ((parallel_generatedRoot_36.w_13_0 ctx), ((parallel_generatedRoot_36.w_14_0 ctx), ((parallel_generatedRoot_36.w_15_0 ctx), ((parallel_generatedRoot_36.w_16_0 ctx), ((parallel_generatedRoot_36.w_17_0 ctx), ((parallel_generatedRoot_36.w_1_0 ctx), ((parallel_generatedRoot_36.w_3_0 ctx), ((parallel_generatedRoot_36.w_5_0 ctx), ((parallel_generatedRoot_36.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_36.w_0_0 ctx), ((parallel_generatedRoot_36.w_0_0 ctx), ((parallel_generatedRoot_36.w_0_0 ctx), ((parallel_generatedRoot_36.w_0_0 ctx), ((parallel_generatedRoot_36.w_0_0 ctx), ((parallel_generatedRoot_36.w_0_0 ctx), ((parallel_generatedRoot_36.w_2_0 ctx), ((parallel_generatedRoot_36.w_4_0 ctx), ((parallel_generatedRoot_36.w_6_0 ctx), ((parallel_generatedRoot_36.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_36.w_18_0 ctx), ((parallel_generatedRoot_36.w_18_1 ctx), ((parallel_generatedRoot_36.w_18_2 ctx), ((parallel_generatedRoot_36.w_18_3 ctx), ((parallel_generatedRoot_36.w_18_4 ctx), ((parallel_generatedRoot_36.w_18_5 ctx), ((parallel_generatedRoot_36.w_18_6 ctx), ((parallel_generatedRoot_36.w_18_7 ctx), ((parallel_generatedRoot_36.w_18_8 ctx), ((parallel_generatedRoot_36.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_36.w_9_0 ctx) (7) (parallel_generatedRoot_36.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_36.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_36.w_10_0 ctx)) (parallel_generatedRoot_36.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_36.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_36.w_10_0 ctx)) (parallel_generatedRoot_36.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_19_0 ctx) (parallel_generatedRoot_36.w_21_0 ctx) (parallel_generatedRoot_36.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_23_concat_1 ctx) (parallel_generatedRoot_36.w_22_0 ctx) (parallel_generatedRoot_36.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_36.w_11_0 ctx) (7) (parallel_generatedRoot_36.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_36.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_36.w_10_0 ctx)) (parallel_generatedRoot_36.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_36.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_36.w_10_0 ctx)) (parallel_generatedRoot_36.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_24_0 ctx) (parallel_generatedRoot_36.w_25_0 ctx) (parallel_generatedRoot_36.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_27_concat_1 ctx) (parallel_generatedRoot_36.w_26_0 ctx) (parallel_generatedRoot_36.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_36.w_23_0 ctx) (parallel_generatedRoot_36.w_27_0 ctx) (parallel_generatedRoot_36.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_36.w_12_0 ctx) (parallel_generatedRoot_36.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_36.w_13_0 ctx) (parallel_generatedRoot_36.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_36.w_14_0 ctx) (parallel_generatedRoot_36.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_29_0 ctx) (parallel_generatedRoot_36.w_30_0 ctx) (parallel_generatedRoot_36.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_32_concat_1 ctx) (parallel_generatedRoot_36.w_31_0 ctx) (parallel_generatedRoot_36.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_36.facts_0  {ctx : parallel_generatedRoot_36Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_36.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_36.w_1_0 ctx) (parallel_generatedRoot_36.w_3_0 ctx) (parallel_generatedRoot_36.w_5_0 ctx) (parallel_generatedRoot_36.w_7_0 ctx) (parallel_generatedRoot_36.w_9_0 ctx) (parallel_generatedRoot_36.w_10_0 ctx) (parallel_generatedRoot_36.w_11_0 ctx) (parallel_generatedRoot_36.w_12_0 ctx) (parallel_generatedRoot_36.w_13_0 ctx) (parallel_generatedRoot_36.w_14_0 ctx) (parallel_generatedRoot_36.w_15_0 ctx) (parallel_generatedRoot_36.w_16_0 ctx) (parallel_generatedRoot_36.w_17_0 ctx) (parallel_generatedRoot_36.w_0_0 ctx) (parallel_generatedRoot_36.w_2_0 ctx) (parallel_generatedRoot_36.w_4_0 ctx) (parallel_generatedRoot_36.w_6_0 ctx) (parallel_generatedRoot_36.w_8_0 ctx) (parallel_generatedRoot_36.w_18_0 ctx) (parallel_generatedRoot_36.w_18_1 ctx) (parallel_generatedRoot_36.w_18_2 ctx) (parallel_generatedRoot_36.w_18_3 ctx) (parallel_generatedRoot_36.w_18_4 ctx) (parallel_generatedRoot_36.w_18_5 ctx) (parallel_generatedRoot_36.w_18_6 ctx) (parallel_generatedRoot_36.w_18_7 ctx) (parallel_generatedRoot_36.w_18_8 ctx) (parallel_generatedRoot_36.w_18_9 ctx) (parallel_generatedRoot_36.w_19_0 ctx) (parallel_generatedRoot_36.w_20_0 ctx) (parallel_generatedRoot_36.w_21_0 ctx) (parallel_generatedRoot_36.w_22_0 ctx) (parallel_generatedRoot_36.w_23_0 ctx) (parallel_generatedRoot_36.w_23_concat_1 ctx) (parallel_generatedRoot_36.w_24_0 ctx) (parallel_generatedRoot_36.w_25_0 ctx) (parallel_generatedRoot_36.w_26_0 ctx) (parallel_generatedRoot_36.w_27_0 ctx) (parallel_generatedRoot_36.w_27_concat_1 ctx) (parallel_generatedRoot_36.w_28_0 ctx) (parallel_generatedRoot_36.w_29_0 ctx) (parallel_generatedRoot_36.w_30_0 ctx) (parallel_generatedRoot_36.w_31_0 ctx) (parallel_generatedRoot_36.w_32_0 ctx) (parallel_generatedRoot_36.w_32_concat_1 ctx)) :
    parallel_generatedRoot_36.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_36.Facts_1  (ctx : parallel_generatedRoot_36Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_36.w_15_0 ctx) (parallel_generatedRoot_36.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_36.w_16_0 ctx) (parallel_generatedRoot_36.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_36.w_17_0 ctx) (parallel_generatedRoot_36.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_33_0 ctx) (parallel_generatedRoot_36.w_34_0 ctx) (parallel_generatedRoot_36.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_36_concat_1 ctx) (parallel_generatedRoot_36.w_35_0 ctx) (parallel_generatedRoot_36.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_36.w_32_0 ctx) (parallel_generatedRoot_36.w_36_0 ctx) (parallel_generatedRoot_36.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_18_6 ctx) (parallel_generatedRoot_36.w_18_8 ctx) (parallel_generatedRoot_36.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_36.w_38_0 ctx) (parallel_generatedRoot_36.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_36.w_39_0 ctx) (parallel_generatedRoot_36.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_36.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_36.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_42_0 ctx) (parallel_generatedRoot_36.w_43_0 ctx) (parallel_generatedRoot_36.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_36.w_44_0 ctx) (parallel_generatedRoot_36.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_36.w_45_0 ctx) (parallel_generatedRoot_36.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_36.facts_1  {ctx : parallel_generatedRoot_36Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_36.constraints_1 backend (parallel_generatedRoot_36.w_15_0 ctx) (parallel_generatedRoot_36.w_16_0 ctx) (parallel_generatedRoot_36.w_17_0 ctx) (parallel_generatedRoot_36.w_18_6 ctx) (parallel_generatedRoot_36.w_18_8 ctx) (parallel_generatedRoot_36.w_32_0 ctx) (parallel_generatedRoot_36.w_33_0 ctx) (parallel_generatedRoot_36.w_34_0 ctx) (parallel_generatedRoot_36.w_35_0 ctx) (parallel_generatedRoot_36.w_36_0 ctx) (parallel_generatedRoot_36.w_36_concat_1 ctx) (parallel_generatedRoot_36.w_37_0 ctx) (parallel_generatedRoot_36.w_38_0 ctx) (parallel_generatedRoot_36.w_39_0 ctx) (parallel_generatedRoot_36.w_40_0 ctx) (parallel_generatedRoot_36.w_41_0 ctx) (parallel_generatedRoot_36.w_42_0 ctx) (parallel_generatedRoot_36.w_43_0 ctx) (parallel_generatedRoot_36.w_44_0 ctx) (parallel_generatedRoot_36.w_45_0 ctx) (parallel_generatedRoot_36.w_46_0 ctx)) :
    parallel_generatedRoot_36.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_36.Facts_2  (ctx : parallel_generatedRoot_36Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_36.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_18_7 ctx) (parallel_generatedRoot_36.w_18_9 ctx) (parallel_generatedRoot_36.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_36.w_50_0 ctx) (parallel_generatedRoot_36.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_36.w_51_0 ctx) (parallel_generatedRoot_36.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_36.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_36.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_36.w_54_0 ctx) (parallel_generatedRoot_36.w_55_0 ctx) (parallel_generatedRoot_36.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_36.w_56_0 ctx) (parallel_generatedRoot_36.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_36.w_57_0 ctx) (parallel_generatedRoot_36.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_36.w_60_0 ctx)

theorem parallel_generatedRoot_36.facts_2  {ctx : parallel_generatedRoot_36Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_36.constraints_2 backend (parallel_generatedRoot_36.w_18_7 ctx) (parallel_generatedRoot_36.w_18_9 ctx) (parallel_generatedRoot_36.w_47_0 ctx) (parallel_generatedRoot_36.w_48_0 ctx) (parallel_generatedRoot_36.w_50_0 ctx) (parallel_generatedRoot_36.w_51_0 ctx) (parallel_generatedRoot_36.w_52_0 ctx) (parallel_generatedRoot_36.w_53_0 ctx) (parallel_generatedRoot_36.w_54_0 ctx) (parallel_generatedRoot_36.w_55_0 ctx) (parallel_generatedRoot_36.w_56_0 ctx) (parallel_generatedRoot_36.w_57_0 ctx) (parallel_generatedRoot_36.w_58_0 ctx) (parallel_generatedRoot_36.w_59_0 ctx) (parallel_generatedRoot_36.w_60_0 ctx)) :
    parallel_generatedRoot_36.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_36.Facts_3  (ctx : parallel_generatedRoot_36Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_36.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_36.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_36.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_36.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_36.facts_3  {ctx : parallel_generatedRoot_36Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_36.constraints_3 (parallel_generatedRoot_36.w_41_0 ctx) (parallel_generatedRoot_36.w_47_0 ctx) (parallel_generatedRoot_36.w_59_0 ctx) (parallel_generatedRoot_36.w_62_0 ctx) (parallel_generatedRoot_36.w_64_0 ctx) (parallel_generatedRoot_36.w_66_0 ctx) (parallel_generatedRoot_36.w_67_0 ctx)) :
    parallel_generatedRoot_36.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_36.Facts_4  (ctx : parallel_generatedRoot_36Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_36.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_36.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_36.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_36.outs ctx

theorem parallel_generatedRoot_36.facts_4  {ctx : parallel_generatedRoot_36Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_36.constraints_4 (parallel_generatedRoot_36.w_49_0 ctx) (parallel_generatedRoot_36.w_53_0 ctx) (parallel_generatedRoot_36.w_59_0 ctx) (parallel_generatedRoot_36.w_61_0 ctx) (parallel_generatedRoot_36.w_63_0 ctx) (parallel_generatedRoot_36.w_65_0 ctx) (parallel_generatedRoot_36.w_69_0 ctx) (parallel_generatedRoot_36.w_70_0 ctx) (parallel_generatedRoot_36.w_71_0 ctx) (parallel_generatedRoot_36.w_73_0 ctx) outputs) :
    parallel_generatedRoot_36.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_36.Facts  (ctx : parallel_generatedRoot_36Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_36.Facts_0 ctx
  c1 : parallel_generatedRoot_36.Facts_1 ctx
  c2 : parallel_generatedRoot_36.Facts_2 ctx
  c3 : parallel_generatedRoot_36.Facts_3 ctx
  c4 : parallel_generatedRoot_36.Facts_4 ctx outputs

theorem parallel_generatedRoot_36.facts  (ctx : parallel_generatedRoot_36Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_36.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_36.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_36.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_36.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_36.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_36.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_36.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_660.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_660.InputType)
  witness : parallel_generatedRoot_660.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_660Ctx" => (parallel_generatedRoot_660.Ctx backend hashModel params i_0)

def parallel_generatedRoot_660.w_0_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 5 → Int :=
  ctx.inputs.1

def parallel_generatedRoot_660.w_16_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_660.w_17_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_660.w_20_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 5 → Int :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_660.w_22_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_660.w_23_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_660.w_24_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_25_0  (ctx : parallel_generatedRoot_660Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_26_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_27_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_28_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_29_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_30_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 5 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_35_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_42_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_47_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_52_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_55_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_58_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_62_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_660.w_1_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  (Int.ofNat i_0)

def parallel_generatedRoot_660.w_2_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  5

def parallel_generatedRoot_660.w_3_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ((parallel_generatedRoot_660.w_1_0 ctx) % (parallel_generatedRoot_660.w_2_0 ctx))

def parallel_generatedRoot_660.w_4_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ctx.witness.w_4_0

def parallel_generatedRoot_660.w_5_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  2

def parallel_generatedRoot_660.w_6_0  (ctx : parallel_generatedRoot_660Ctx) : Bool :=
  decide ((parallel_generatedRoot_660.w_4_0 ctx) < (parallel_generatedRoot_660.w_5_0 ctx))

def parallel_generatedRoot_660.w_7_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  if (parallel_generatedRoot_660.w_6_0 ctx) then 1 else 0

def parallel_generatedRoot_660.w_8_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  2

def parallel_generatedRoot_660.w_9_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  0

def parallel_generatedRoot_660.w_10_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ((parallel_generatedRoot_660.w_8_0 ctx) + (parallel_generatedRoot_660.w_9_0 ctx))

def parallel_generatedRoot_660.w_11_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ctx.witness.w_11_0

def parallel_generatedRoot_660.w_12_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  1

def parallel_generatedRoot_660.w_13_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ((parallel_generatedRoot_660.w_11_0 ctx) + (parallel_generatedRoot_660.w_12_0 ctx))

def parallel_generatedRoot_660.w_14_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  2

def parallel_generatedRoot_660.w_15_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ((parallel_generatedRoot_660.w_13_0 ctx) / (parallel_generatedRoot_660.w_14_0 ctx))

def parallel_generatedRoot_660.w_18_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  5

def parallel_generatedRoot_660.w_19_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ((parallel_generatedRoot_660.w_1_0 ctx) / (parallel_generatedRoot_660.w_18_0 ctx))

def parallel_generatedRoot_660.w_21_0  (ctx : parallel_generatedRoot_660Ctx) : Int :=
  ctx.witness.w_21_0

def parallel_generatedRoot_660.w_31_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_660.w_31_1  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_1

def parallel_generatedRoot_660.w_31_2  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_2

def parallel_generatedRoot_660.w_31_3  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_3

def parallel_generatedRoot_660.w_31_4  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_4

def parallel_generatedRoot_660.w_31_5  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_5

def parallel_generatedRoot_660.w_31_6  (ctx : parallel_generatedRoot_660Ctx) : Fin 4 → mxxMat5 :=
  ctx.witness.w_31_6

def parallel_generatedRoot_660.w_32_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_660.w_32_1  (ctx : parallel_generatedRoot_660Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_1

def parallel_generatedRoot_660.w_32_2  (ctx : parallel_generatedRoot_660Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_2

def parallel_generatedRoot_660.w_32_3  (ctx : parallel_generatedRoot_660Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_3

def parallel_generatedRoot_660.w_32_4  (ctx : parallel_generatedRoot_660Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_4

def parallel_generatedRoot_660.w_32_5  (ctx : parallel_generatedRoot_660Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_5

def parallel_generatedRoot_660.w_32_6  (ctx : parallel_generatedRoot_660Ctx) : Fin 2 → mxxMat5 :=
  ctx.witness.w_32_6

def parallel_generatedRoot_660.w_33_0  (ctx : parallel_generatedRoot_660Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_660.w_33_1  (ctx : parallel_generatedRoot_660Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_1

def parallel_generatedRoot_660.w_33_2  (ctx : parallel_generatedRoot_660Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_2

def parallel_generatedRoot_660.w_33_3  (ctx : parallel_generatedRoot_660Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_3

def parallel_generatedRoot_660.w_33_4  (ctx : parallel_generatedRoot_660Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_4

def parallel_generatedRoot_660.w_33_5  (ctx : parallel_generatedRoot_660Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_5

def parallel_generatedRoot_660.w_33_6  (ctx : parallel_generatedRoot_660Ctx) : Fin 1 → mxxMat5 :=
  ctx.witness.w_33_6

def parallel_generatedRoot_660.w_34_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_660.w_36_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_660.w_37_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_660.w_38_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_660.w_36_0 ctx) (parallel_generatedRoot_660.w_37_0 ctx)

def parallel_generatedRoot_660.w_39_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat6 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_660.w_40_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat6 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_660.w_41_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_34_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx)

def parallel_generatedRoot_660.w_43_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat6 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_660.w_44_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_41_0 ctx) (parallel_generatedRoot_660.w_43_0 ctx)

def parallel_generatedRoot_660.w_45_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_660.w_46_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_45_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx)

def parallel_generatedRoot_660.w_48_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat6 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_660.w_49_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_46_0 ctx) (parallel_generatedRoot_660.w_48_0 ctx)

def parallel_generatedRoot_660.w_50_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_44_0 ctx) (parallel_generatedRoot_660.w_49_0 ctx)

def parallel_generatedRoot_660.w_51_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat5 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_660.w_53_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_660.w_51_0 ctx) (parallel_generatedRoot_660.w_52_0 ctx)

def parallel_generatedRoot_660.w_54_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_50_0 ctx) (parallel_generatedRoot_660.w_53_0 ctx)

def parallel_generatedRoot_660.w_56_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat6 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_660.w_57_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_41_0 ctx) (parallel_generatedRoot_660.w_56_0 ctx)

def parallel_generatedRoot_660.w_59_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat6 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_660.w_60_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_46_0 ctx) (parallel_generatedRoot_660.w_59_0 ctx)

def parallel_generatedRoot_660.w_61_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_57_0 ctx) (parallel_generatedRoot_660.w_60_0 ctx)

def parallel_generatedRoot_660.w_63_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_660.w_51_0 ctx) (parallel_generatedRoot_660.w_62_0 ctx)

def parallel_generatedRoot_660.w_64_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_61_0 ctx) (parallel_generatedRoot_660.w_63_0 ctx)

def parallel_generatedRoot_660.w_65_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.witness.w_65_0

def parallel_generatedRoot_660.w_66_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_65_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx)

def parallel_generatedRoot_660.w_67_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_66_0 ctx) (parallel_generatedRoot_660.w_43_0 ctx)

def parallel_generatedRoot_660.w_68_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.witness.w_68_0

def parallel_generatedRoot_660.w_69_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_68_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx)

def parallel_generatedRoot_660.w_70_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_69_0 ctx) (parallel_generatedRoot_660.w_48_0 ctx)

def parallel_generatedRoot_660.w_71_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_67_0 ctx) (parallel_generatedRoot_660.w_70_0 ctx)

def parallel_generatedRoot_660.w_72_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_66_0 ctx) (parallel_generatedRoot_660.w_56_0 ctx)

def parallel_generatedRoot_660.w_73_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_69_0 ctx) (parallel_generatedRoot_660.w_59_0 ctx)

def parallel_generatedRoot_660.w_74_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_72_0 ctx) (parallel_generatedRoot_660.w_73_0 ctx)

def parallel_generatedRoot_660.w_75_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_660.w_76_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_75_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx)

def parallel_generatedRoot_660.w_77_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_76_0 ctx) (parallel_generatedRoot_660.w_43_0 ctx)

def parallel_generatedRoot_660.w_78_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  ctx.witness.w_78_0

def parallel_generatedRoot_660.w_79_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_78_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx)

def parallel_generatedRoot_660.w_80_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_79_0 ctx) (parallel_generatedRoot_660.w_48_0 ctx)

def parallel_generatedRoot_660.w_81_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_77_0 ctx) (parallel_generatedRoot_660.w_80_0 ctx)

def parallel_generatedRoot_660.w_82_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_76_0 ctx) (parallel_generatedRoot_660.w_56_0 ctx)

def parallel_generatedRoot_660.w_83_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_660.w_79_0 ctx) (parallel_generatedRoot_660.w_59_0 ctx)

def parallel_generatedRoot_660.w_84_0  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_660.w_82_0 ctx) (parallel_generatedRoot_660.w_83_0 ctx)

def parallel_generatedRoot_660.outs  (ctx : parallel_generatedRoot_660Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_660.w_54_0 ctx), ((parallel_generatedRoot_660.w_64_0 ctx), ((parallel_generatedRoot_660.w_71_0 ctx), ((parallel_generatedRoot_660.w_74_0 ctx), ((parallel_generatedRoot_660.w_81_0 ctx), ((parallel_generatedRoot_660.w_84_0 ctx), ()))))))

structure parallel_generatedRoot_660.Facts_0  (ctx : parallel_generatedRoot_660Ctx) : Prop where
  h2_0 : (parallel_generatedRoot_660.w_2_0 ctx) ≠ 0
  h3_0 : 0 ≤ (parallel_generatedRoot_660.w_3_0 ctx)
  h3_0_1 : (parallel_generatedRoot_660.w_3_0 ctx) < 5
  h4_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_660.w_0_0 ctx) (parallel_generatedRoot_660.w_3_0 ctx) (parallel_generatedRoot_660.w_4_0 ctx)
  h7_0 : 0 ≤ (parallel_generatedRoot_660.w_7_0 ctx)
  h7_0_1 : (parallel_generatedRoot_660.w_7_0 ctx) < 2
  t6 : 2 = 2
  h11_0 : MxxRuntime.select (parallel_generatedRoot_660.w_7_0 ctx) [(parallel_generatedRoot_660.w_4_0 ctx), (parallel_generatedRoot_660.w_10_0 ctx)] (parallel_generatedRoot_660.w_11_0 ctx)
  h14_0 : (parallel_generatedRoot_660.w_14_0 ctx) ≠ 0
  h18_0 : (parallel_generatedRoot_660.w_18_0 ctx) ≠ 0
  h3_0_2 : 0 ≤ (parallel_generatedRoot_660.w_3_0 ctx)
  h3_0_3 : (parallel_generatedRoot_660.w_3_0 ctx) < 5
  h21_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_660.w_20_0 ctx) (parallel_generatedRoot_660.w_3_0 ctx) (parallel_generatedRoot_660.w_21_0 ctx)
  t13 : (4) = 4
  h31_6 : (∀ i : Fin 4, parallel_parallel_generatedRoot_660_31 backend hashModel params i_0 i ((parallel_generatedRoot_660.w_4_0 ctx), ((parallel_generatedRoot_660.w_16_0 ctx), ((parallel_generatedRoot_660.w_17_0 ctx), ((parallel_generatedRoot_660.w_19_0 ctx), ((parallel_generatedRoot_660.w_21_0 ctx), ((parallel_generatedRoot_660.w_22_0 ctx), ((parallel_generatedRoot_660.w_23_0 ctx), ((parallel_generatedRoot_660.w_24_0 ctx), ((parallel_generatedRoot_660.w_25_0 ctx), ((parallel_generatedRoot_660.w_26_0 ctx), ((parallel_generatedRoot_660.w_27_0 ctx), ((parallel_generatedRoot_660.w_28_0 ctx), ((parallel_generatedRoot_660.w_29_0 ctx), ((parallel_generatedRoot_660.w_30_0 ctx), ())))))))))))))) (((parallel_generatedRoot_660.w_31_0 ctx) i), (((parallel_generatedRoot_660.w_31_1 ctx) i), (((parallel_generatedRoot_660.w_31_2 ctx) i), (((parallel_generatedRoot_660.w_31_3 ctx) i), (((parallel_generatedRoot_660.w_31_4 ctx) i), (((parallel_generatedRoot_660.w_31_5 ctx) i), (((parallel_generatedRoot_660.w_31_6 ctx) i), ()))))))))
  t15 : (2) = 2
  h32_6 : (∀ i : Fin 2, parallel_parallel_generatedRoot_660_32 backend hashModel params i_0 i ((parallel_generatedRoot_660.w_11_0 ctx), ((parallel_generatedRoot_660.w_31_0 ctx), ((parallel_generatedRoot_660.w_31_2 ctx), ((parallel_generatedRoot_660.w_31_1 ctx), ((parallel_generatedRoot_660.w_31_4 ctx), ((parallel_generatedRoot_660.w_31_6 ctx), ((parallel_generatedRoot_660.w_31_3 ctx), ((parallel_generatedRoot_660.w_31_5 ctx), ())))))))) (((parallel_generatedRoot_660.w_32_0 ctx) i), (((parallel_generatedRoot_660.w_32_1 ctx) i), (((parallel_generatedRoot_660.w_32_2 ctx) i), (((parallel_generatedRoot_660.w_32_3 ctx) i), (((parallel_generatedRoot_660.w_32_4 ctx) i), (((parallel_generatedRoot_660.w_32_5 ctx) i), (((parallel_generatedRoot_660.w_32_6 ctx) i), ()))))))))
  t17 : (1) = 1
  h33_6 : (∀ i : Fin 1, parallel_parallel_generatedRoot_660_33 backend hashModel params i_0 i ((parallel_generatedRoot_660.w_15_0 ctx), ((parallel_generatedRoot_660.w_32_0 ctx), ((parallel_generatedRoot_660.w_32_2 ctx), ((parallel_generatedRoot_660.w_32_1 ctx), ((parallel_generatedRoot_660.w_32_4 ctx), ((parallel_generatedRoot_660.w_32_6 ctx), ((parallel_generatedRoot_660.w_32_3 ctx), ((parallel_generatedRoot_660.w_32_5 ctx), ())))))))) (((parallel_generatedRoot_660.w_33_0 ctx) i), (((parallel_generatedRoot_660.w_33_1 ctx) i), (((parallel_generatedRoot_660.w_33_2 ctx) i), (((parallel_generatedRoot_660.w_33_3 ctx) i), (((parallel_generatedRoot_660.w_33_4 ctx) i), (((parallel_generatedRoot_660.w_33_5 ctx) i), (((parallel_generatedRoot_660.w_33_6 ctx) i), ()))))))))
  h34_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_660.w_33_0 ctx) (0) (parallel_generatedRoot_660.w_34_0 ctx)
  h36_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_660.w_35_0 ctx) (parallel_generatedRoot_660.w_36_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_660.w_38_0 ctx) (parallel_generatedRoot_660.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_660.w_39_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx)
  h43_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_660.w_42_0 ctx) (parallel_generatedRoot_660.w_43_0 ctx)
  h45_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_660.w_33_1 ctx) (0) (parallel_generatedRoot_660.w_45_0 ctx)
  h48_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_660.w_47_0 ctx) (parallel_generatedRoot_660.w_48_0 ctx)
  h51_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_660.w_33_6 ctx) (0) (parallel_generatedRoot_660.w_51_0 ctx)
  h56_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_660.w_55_0 ctx) (parallel_generatedRoot_660.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_660.w_58_0 ctx) (parallel_generatedRoot_660.w_59_0 ctx)
  h65_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_660.w_33_2 ctx) (0) (parallel_generatedRoot_660.w_65_0 ctx)
  h68_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_660.w_33_3 ctx) (0) (parallel_generatedRoot_660.w_68_0 ctx)
  h75_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_660.w_33_4 ctx) (0) (parallel_generatedRoot_660.w_75_0 ctx)

theorem parallel_generatedRoot_660.facts_0  {ctx : parallel_generatedRoot_660Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_660.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_660.w_0_0 ctx) (parallel_generatedRoot_660.w_16_0 ctx) (parallel_generatedRoot_660.w_17_0 ctx) (parallel_generatedRoot_660.w_20_0 ctx) (parallel_generatedRoot_660.w_22_0 ctx) (parallel_generatedRoot_660.w_23_0 ctx) (parallel_generatedRoot_660.w_24_0 ctx) (parallel_generatedRoot_660.w_25_0 ctx) (parallel_generatedRoot_660.w_26_0 ctx) (parallel_generatedRoot_660.w_27_0 ctx) (parallel_generatedRoot_660.w_28_0 ctx) (parallel_generatedRoot_660.w_29_0 ctx) (parallel_generatedRoot_660.w_30_0 ctx) (parallel_generatedRoot_660.w_35_0 ctx) (parallel_generatedRoot_660.w_42_0 ctx) (parallel_generatedRoot_660.w_47_0 ctx) (parallel_generatedRoot_660.w_55_0 ctx) (parallel_generatedRoot_660.w_58_0 ctx) (parallel_generatedRoot_660.w_2_0 ctx) (parallel_generatedRoot_660.w_3_0 ctx) (parallel_generatedRoot_660.w_4_0 ctx) (parallel_generatedRoot_660.w_7_0 ctx) (parallel_generatedRoot_660.w_10_0 ctx) (parallel_generatedRoot_660.w_11_0 ctx) (parallel_generatedRoot_660.w_14_0 ctx) (parallel_generatedRoot_660.w_15_0 ctx) (parallel_generatedRoot_660.w_18_0 ctx) (parallel_generatedRoot_660.w_19_0 ctx) (parallel_generatedRoot_660.w_21_0 ctx) (parallel_generatedRoot_660.w_31_0 ctx) (parallel_generatedRoot_660.w_31_1 ctx) (parallel_generatedRoot_660.w_31_2 ctx) (parallel_generatedRoot_660.w_31_3 ctx) (parallel_generatedRoot_660.w_31_4 ctx) (parallel_generatedRoot_660.w_31_5 ctx) (parallel_generatedRoot_660.w_31_6 ctx) (parallel_generatedRoot_660.w_32_0 ctx) (parallel_generatedRoot_660.w_32_1 ctx) (parallel_generatedRoot_660.w_32_2 ctx) (parallel_generatedRoot_660.w_32_3 ctx) (parallel_generatedRoot_660.w_32_4 ctx) (parallel_generatedRoot_660.w_32_5 ctx) (parallel_generatedRoot_660.w_32_6 ctx) (parallel_generatedRoot_660.w_33_0 ctx) (parallel_generatedRoot_660.w_33_1 ctx) (parallel_generatedRoot_660.w_33_2 ctx) (parallel_generatedRoot_660.w_33_3 ctx) (parallel_generatedRoot_660.w_33_4 ctx) (parallel_generatedRoot_660.w_33_5 ctx) (parallel_generatedRoot_660.w_33_6 ctx) (parallel_generatedRoot_660.w_34_0 ctx) (parallel_generatedRoot_660.w_36_0 ctx) (parallel_generatedRoot_660.w_38_0 ctx) (parallel_generatedRoot_660.w_39_0 ctx) (parallel_generatedRoot_660.w_40_0 ctx) (parallel_generatedRoot_660.w_43_0 ctx) (parallel_generatedRoot_660.w_45_0 ctx) (parallel_generatedRoot_660.w_48_0 ctx) (parallel_generatedRoot_660.w_51_0 ctx) (parallel_generatedRoot_660.w_56_0 ctx) (parallel_generatedRoot_660.w_59_0 ctx) (parallel_generatedRoot_660.w_65_0 ctx) (parallel_generatedRoot_660.w_68_0 ctx) (parallel_generatedRoot_660.w_75_0 ctx)) :
    parallel_generatedRoot_660.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_660.Facts_1  (ctx : parallel_generatedRoot_660Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  h78_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_660.w_33_5 ctx) (0) (parallel_generatedRoot_660.w_78_0 ctx)
  hout : outputs = parallel_generatedRoot_660.outs ctx

theorem parallel_generatedRoot_660.facts_1  {ctx : parallel_generatedRoot_660Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_660.constraints_1 (parallel_generatedRoot_660.w_33_5 ctx) (parallel_generatedRoot_660.w_54_0 ctx) (parallel_generatedRoot_660.w_64_0 ctx) (parallel_generatedRoot_660.w_71_0 ctx) (parallel_generatedRoot_660.w_74_0 ctx) (parallel_generatedRoot_660.w_78_0 ctx) (parallel_generatedRoot_660.w_81_0 ctx) (parallel_generatedRoot_660.w_84_0 ctx) outputs) :
    parallel_generatedRoot_660.Facts_1 ctx outputs :=
  ⟨h.1, h.2⟩

structure parallel_generatedRoot_660.Facts  (ctx : parallel_generatedRoot_660Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_660.Facts_0 ctx
  c1 : parallel_generatedRoot_660.Facts_1 ctx outputs

theorem parallel_generatedRoot_660.facts  (ctx : parallel_generatedRoot_660Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_660.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_660.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_660.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_660.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_892.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit
  witness : parallel_generatedRoot_892.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_892Ctx" => (parallel_generatedRoot_892.Ctx backend hashModel params i_0)

def parallel_generatedRoot_892.w_0_0  (ctx : parallel_generatedRoot_892Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_892.w_1_0  (ctx : parallel_generatedRoot_892Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_892.w_2_0  (ctx : parallel_generatedRoot_892Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_892.w_3_0  (ctx : parallel_generatedRoot_892Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_892.w_4_0  (ctx : parallel_generatedRoot_892Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_892.w_5_0  (ctx : parallel_generatedRoot_892Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_892.w_6_0  (ctx : parallel_generatedRoot_892Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_892.w_6_1  (ctx : parallel_generatedRoot_892Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_892.w_6_2  (ctx : parallel_generatedRoot_892Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_892.w_6_3  (ctx : parallel_generatedRoot_892Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_892.w_6_4  (ctx : parallel_generatedRoot_892Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_892.w_6_5  (ctx : parallel_generatedRoot_892Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_892.w_7_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_892.w_8_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_892.w_9_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_892.w_7_0 ctx) (parallel_generatedRoot_892.w_8_0 ctx)

def parallel_generatedRoot_892.w_10_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_892.w_11_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_892.w_12_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_892.w_10_0 ctx) (parallel_generatedRoot_892.w_11_0 ctx)

def parallel_generatedRoot_892.w_13_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_892.w_14_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_892.w_15_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_892.w_13_0 ctx) (parallel_generatedRoot_892.w_14_0 ctx)

def parallel_generatedRoot_892.w_16_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_892.w_17_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_892.w_18_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_892.w_16_0 ctx) (parallel_generatedRoot_892.w_17_0 ctx)

def parallel_generatedRoot_892.w_19_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_892.w_20_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_892.w_21_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_892.w_19_0 ctx) (parallel_generatedRoot_892.w_20_0 ctx)

def parallel_generatedRoot_892.w_22_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_892.w_23_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_892.w_24_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_892.w_22_0 ctx) (parallel_generatedRoot_892.w_23_0 ctx)

def parallel_generatedRoot_892.w_25_0  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_892.w_25_1  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_892.w_25_2  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_892.w_25_3  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_892.w_25_4  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_892.w_25_5  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_892.outs  (ctx : parallel_generatedRoot_892Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_892.w_25_0 ctx), ((parallel_generatedRoot_892.w_25_1 ctx), ((parallel_generatedRoot_892.w_25_2 ctx), ((parallel_generatedRoot_892.w_25_3 ctx), ((parallel_generatedRoot_892.w_25_4 ctx), ((parallel_generatedRoot_892.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_892.Facts_0  (ctx : parallel_generatedRoot_892Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_892_6 backend hashModel params i_0 i ((parallel_generatedRoot_892.w_0_0 ctx), ((parallel_generatedRoot_892.w_1_0 ctx), ((parallel_generatedRoot_892.w_2_0 ctx), ((parallel_generatedRoot_892.w_3_0 ctx), ((parallel_generatedRoot_892.w_4_0 ctx), ((parallel_generatedRoot_892.w_5_0 ctx), ())))))) (((parallel_generatedRoot_892.w_6_0 ctx) i), (((parallel_generatedRoot_892.w_6_1 ctx) i), (((parallel_generatedRoot_892.w_6_2 ctx) i), (((parallel_generatedRoot_892.w_6_3 ctx) i), (((parallel_generatedRoot_892.w_6_4 ctx) i), (((parallel_generatedRoot_892.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_0 ctx) (0) (parallel_generatedRoot_892.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_0 ctx) (1) (parallel_generatedRoot_892.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_1 ctx) (0) (parallel_generatedRoot_892.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_1 ctx) (1) (parallel_generatedRoot_892.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_2 ctx) (0) (parallel_generatedRoot_892.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_2 ctx) (1) (parallel_generatedRoot_892.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_3 ctx) (0) (parallel_generatedRoot_892.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_3 ctx) (1) (parallel_generatedRoot_892.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_4 ctx) (0) (parallel_generatedRoot_892.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_4 ctx) (1) (parallel_generatedRoot_892.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_5 ctx) (0) (parallel_generatedRoot_892.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_892.w_6_5 ctx) (1) (parallel_generatedRoot_892.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_892_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_892.w_6_0 ctx), ((parallel_generatedRoot_892.w_6_1 ctx), ((parallel_generatedRoot_892.w_6_2 ctx), ((parallel_generatedRoot_892.w_6_3 ctx), ((parallel_generatedRoot_892.w_6_4 ctx), ((parallel_generatedRoot_892.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_892.w_9_0 ctx), ((parallel_generatedRoot_892.w_12_0 ctx), ((parallel_generatedRoot_892.w_15_0 ctx), ((parallel_generatedRoot_892.w_18_0 ctx), ((parallel_generatedRoot_892.w_21_0 ctx), ((parallel_generatedRoot_892.w_24_0 ctx), ())))))) ((parallel_generatedRoot_892.w_25_0 ctx), ((parallel_generatedRoot_892.w_25_1 ctx), ((parallel_generatedRoot_892.w_25_2 ctx), ((parallel_generatedRoot_892.w_25_3 ctx), ((parallel_generatedRoot_892.w_25_4 ctx), ((parallel_generatedRoot_892.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_892.outs ctx

theorem parallel_generatedRoot_892.facts_0  {ctx : parallel_generatedRoot_892Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_892.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_892.w_0_0 ctx) (parallel_generatedRoot_892.w_1_0 ctx) (parallel_generatedRoot_892.w_2_0 ctx) (parallel_generatedRoot_892.w_3_0 ctx) (parallel_generatedRoot_892.w_4_0 ctx) (parallel_generatedRoot_892.w_5_0 ctx) (parallel_generatedRoot_892.w_6_0 ctx) (parallel_generatedRoot_892.w_6_1 ctx) (parallel_generatedRoot_892.w_6_2 ctx) (parallel_generatedRoot_892.w_6_3 ctx) (parallel_generatedRoot_892.w_6_4 ctx) (parallel_generatedRoot_892.w_6_5 ctx) (parallel_generatedRoot_892.w_7_0 ctx) (parallel_generatedRoot_892.w_8_0 ctx) (parallel_generatedRoot_892.w_9_0 ctx) (parallel_generatedRoot_892.w_10_0 ctx) (parallel_generatedRoot_892.w_11_0 ctx) (parallel_generatedRoot_892.w_12_0 ctx) (parallel_generatedRoot_892.w_13_0 ctx) (parallel_generatedRoot_892.w_14_0 ctx) (parallel_generatedRoot_892.w_15_0 ctx) (parallel_generatedRoot_892.w_16_0 ctx) (parallel_generatedRoot_892.w_17_0 ctx) (parallel_generatedRoot_892.w_18_0 ctx) (parallel_generatedRoot_892.w_19_0 ctx) (parallel_generatedRoot_892.w_20_0 ctx) (parallel_generatedRoot_892.w_21_0 ctx) (parallel_generatedRoot_892.w_22_0 ctx) (parallel_generatedRoot_892.w_23_0 ctx) (parallel_generatedRoot_892.w_24_0 ctx) (parallel_generatedRoot_892.w_25_0 ctx) (parallel_generatedRoot_892.w_25_1 ctx) (parallel_generatedRoot_892.w_25_2 ctx) (parallel_generatedRoot_892.w_25_3 ctx) (parallel_generatedRoot_892.w_25_4 ctx) (parallel_generatedRoot_892.w_25_5 ctx) outputs) :
    parallel_generatedRoot_892.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_892.Facts  (ctx : parallel_generatedRoot_892Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_892.Facts_0 ctx outputs

theorem parallel_generatedRoot_892.facts  (ctx : parallel_generatedRoot_892Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_892.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_892.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_892.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_896.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit
  witness : parallel_generatedRoot_896.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_896Ctx" => (parallel_generatedRoot_896.Ctx backend hashModel params i_0)

def parallel_generatedRoot_896.w_0_0  (ctx : parallel_generatedRoot_896Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_896.w_1_0  (ctx : parallel_generatedRoot_896Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_896.w_2_0  (ctx : parallel_generatedRoot_896Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_896.w_3_0  (ctx : parallel_generatedRoot_896Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_896.w_4_0  (ctx : parallel_generatedRoot_896Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_896.w_5_0  (ctx : parallel_generatedRoot_896Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_896.w_6_0  (ctx : parallel_generatedRoot_896Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_896.w_6_1  (ctx : parallel_generatedRoot_896Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_896.w_6_2  (ctx : parallel_generatedRoot_896Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_896.w_6_3  (ctx : parallel_generatedRoot_896Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_896.w_6_4  (ctx : parallel_generatedRoot_896Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_896.w_6_5  (ctx : parallel_generatedRoot_896Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_896.w_7_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_896.w_8_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_896.w_9_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_896.w_7_0 ctx) (parallel_generatedRoot_896.w_8_0 ctx)

def parallel_generatedRoot_896.w_10_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_896.w_11_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_896.w_12_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_896.w_10_0 ctx) (parallel_generatedRoot_896.w_11_0 ctx)

def parallel_generatedRoot_896.w_13_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_896.w_14_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_896.w_15_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_896.w_13_0 ctx) (parallel_generatedRoot_896.w_14_0 ctx)

def parallel_generatedRoot_896.w_16_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_896.w_17_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_896.w_18_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_896.w_16_0 ctx) (parallel_generatedRoot_896.w_17_0 ctx)

def parallel_generatedRoot_896.w_19_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_896.w_20_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_896.w_21_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_896.w_19_0 ctx) (parallel_generatedRoot_896.w_20_0 ctx)

def parallel_generatedRoot_896.w_22_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_896.w_23_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_896.w_24_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_896.w_22_0 ctx) (parallel_generatedRoot_896.w_23_0 ctx)

def parallel_generatedRoot_896.w_25_0  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_896.w_25_1  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_896.w_25_2  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_896.w_25_3  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_896.w_25_4  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_896.w_25_5  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_896.outs  (ctx : parallel_generatedRoot_896Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_896.w_25_0 ctx), ((parallel_generatedRoot_896.w_25_1 ctx), ((parallel_generatedRoot_896.w_25_2 ctx), ((parallel_generatedRoot_896.w_25_3 ctx), ((parallel_generatedRoot_896.w_25_4 ctx), ((parallel_generatedRoot_896.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_896.Facts_0  (ctx : parallel_generatedRoot_896Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_896_6 backend hashModel params i_0 i ((parallel_generatedRoot_896.w_0_0 ctx), ((parallel_generatedRoot_896.w_1_0 ctx), ((parallel_generatedRoot_896.w_2_0 ctx), ((parallel_generatedRoot_896.w_3_0 ctx), ((parallel_generatedRoot_896.w_4_0 ctx), ((parallel_generatedRoot_896.w_5_0 ctx), ())))))) (((parallel_generatedRoot_896.w_6_0 ctx) i), (((parallel_generatedRoot_896.w_6_1 ctx) i), (((parallel_generatedRoot_896.w_6_2 ctx) i), (((parallel_generatedRoot_896.w_6_3 ctx) i), (((parallel_generatedRoot_896.w_6_4 ctx) i), (((parallel_generatedRoot_896.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_0 ctx) (0) (parallel_generatedRoot_896.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_0 ctx) (1) (parallel_generatedRoot_896.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_1 ctx) (0) (parallel_generatedRoot_896.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_1 ctx) (1) (parallel_generatedRoot_896.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_2 ctx) (0) (parallel_generatedRoot_896.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_2 ctx) (1) (parallel_generatedRoot_896.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_3 ctx) (0) (parallel_generatedRoot_896.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_3 ctx) (1) (parallel_generatedRoot_896.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_4 ctx) (0) (parallel_generatedRoot_896.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_4 ctx) (1) (parallel_generatedRoot_896.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_5 ctx) (0) (parallel_generatedRoot_896.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_896.w_6_5 ctx) (1) (parallel_generatedRoot_896.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_896_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_896.w_6_0 ctx), ((parallel_generatedRoot_896.w_6_1 ctx), ((parallel_generatedRoot_896.w_6_2 ctx), ((parallel_generatedRoot_896.w_6_3 ctx), ((parallel_generatedRoot_896.w_6_4 ctx), ((parallel_generatedRoot_896.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_896.w_9_0 ctx), ((parallel_generatedRoot_896.w_12_0 ctx), ((parallel_generatedRoot_896.w_15_0 ctx), ((parallel_generatedRoot_896.w_18_0 ctx), ((parallel_generatedRoot_896.w_21_0 ctx), ((parallel_generatedRoot_896.w_24_0 ctx), ())))))) ((parallel_generatedRoot_896.w_25_0 ctx), ((parallel_generatedRoot_896.w_25_1 ctx), ((parallel_generatedRoot_896.w_25_2 ctx), ((parallel_generatedRoot_896.w_25_3 ctx), ((parallel_generatedRoot_896.w_25_4 ctx), ((parallel_generatedRoot_896.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_896.outs ctx

theorem parallel_generatedRoot_896.facts_0  {ctx : parallel_generatedRoot_896Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_896.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_896.w_0_0 ctx) (parallel_generatedRoot_896.w_1_0 ctx) (parallel_generatedRoot_896.w_2_0 ctx) (parallel_generatedRoot_896.w_3_0 ctx) (parallel_generatedRoot_896.w_4_0 ctx) (parallel_generatedRoot_896.w_5_0 ctx) (parallel_generatedRoot_896.w_6_0 ctx) (parallel_generatedRoot_896.w_6_1 ctx) (parallel_generatedRoot_896.w_6_2 ctx) (parallel_generatedRoot_896.w_6_3 ctx) (parallel_generatedRoot_896.w_6_4 ctx) (parallel_generatedRoot_896.w_6_5 ctx) (parallel_generatedRoot_896.w_7_0 ctx) (parallel_generatedRoot_896.w_8_0 ctx) (parallel_generatedRoot_896.w_9_0 ctx) (parallel_generatedRoot_896.w_10_0 ctx) (parallel_generatedRoot_896.w_11_0 ctx) (parallel_generatedRoot_896.w_12_0 ctx) (parallel_generatedRoot_896.w_13_0 ctx) (parallel_generatedRoot_896.w_14_0 ctx) (parallel_generatedRoot_896.w_15_0 ctx) (parallel_generatedRoot_896.w_16_0 ctx) (parallel_generatedRoot_896.w_17_0 ctx) (parallel_generatedRoot_896.w_18_0 ctx) (parallel_generatedRoot_896.w_19_0 ctx) (parallel_generatedRoot_896.w_20_0 ctx) (parallel_generatedRoot_896.w_21_0 ctx) (parallel_generatedRoot_896.w_22_0 ctx) (parallel_generatedRoot_896.w_23_0 ctx) (parallel_generatedRoot_896.w_24_0 ctx) (parallel_generatedRoot_896.w_25_0 ctx) (parallel_generatedRoot_896.w_25_1 ctx) (parallel_generatedRoot_896.w_25_2 ctx) (parallel_generatedRoot_896.w_25_3 ctx) (parallel_generatedRoot_896.w_25_4 ctx) (parallel_generatedRoot_896.w_25_5 ctx) outputs) :
    parallel_generatedRoot_896.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_896.Facts  (ctx : parallel_generatedRoot_896Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_896.Facts_0 ctx outputs

theorem parallel_generatedRoot_896.facts  (ctx : parallel_generatedRoot_896Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_896.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_896.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_896.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_897.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_897.InputType)
  witness : parallel_generatedRoot_897.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_897Ctx" => (parallel_generatedRoot_897.Ctx backend hashModel params i_0)

def parallel_generatedRoot_897.w_1_0  (ctx : parallel_generatedRoot_897Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_897.w_3_0  (ctx : parallel_generatedRoot_897Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_897.w_5_0  (ctx : parallel_generatedRoot_897Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_897.w_7_0  (ctx : parallel_generatedRoot_897Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_897.w_9_0  (ctx : parallel_generatedRoot_897Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_897.w_10_0  (ctx : parallel_generatedRoot_897Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_897.w_11_0  (ctx : parallel_generatedRoot_897Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_897.w_12_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_897.w_13_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_897.w_14_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_897.w_15_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_897.w_16_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_897.w_17_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_897.w_0_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_897.w_2_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_897.w_4_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_897.w_6_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_897.w_8_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_897.w_18_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_897.w_18_1  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_897.w_18_2  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_897.w_18_3  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_897.w_18_4  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_897.w_18_5  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_897.w_18_6  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_897.w_18_7  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_897.w_18_8  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_897.w_18_9  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_897.w_19_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_897.w_20_0  (ctx : parallel_generatedRoot_897Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_897.w_21_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_897.w_22_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_897.w_23_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_897.w_23_concat_1  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_897.w_24_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_897.w_25_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_897.w_26_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_897.w_27_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_897.w_27_concat_1  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_897.w_28_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_897.w_29_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_897.w_30_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_897.w_31_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_897.w_32_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_897.w_32_concat_1  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_897.w_33_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_897.w_34_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_897.w_35_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_897.w_36_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_897.w_36_concat_1  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_897.w_37_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_897.w_38_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_897.w_39_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_897.w_40_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_897.w_41_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_897.w_37_0 ctx) (parallel_generatedRoot_897.w_40_0 ctx)

def parallel_generatedRoot_897.w_42_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_897.w_43_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_897.w_44_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_897.w_45_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_897.w_46_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_897.w_47_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_897.w_28_0 ctx) (parallel_generatedRoot_897.w_46_0 ctx)

def parallel_generatedRoot_897.w_48_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_897.w_49_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_18_2 ctx) (parallel_generatedRoot_897.w_48_0 ctx)

def parallel_generatedRoot_897.w_50_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_897.w_51_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_897.w_52_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_897.w_53_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_897.w_37_0 ctx) (parallel_generatedRoot_897.w_52_0 ctx)

def parallel_generatedRoot_897.w_54_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_897.w_55_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_897.w_56_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_897.w_57_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_897.w_58_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_897.w_59_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_897.w_28_0 ctx) (parallel_generatedRoot_897.w_58_0 ctx)

def parallel_generatedRoot_897.w_60_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_897.w_61_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_18_3 ctx) (parallel_generatedRoot_897.w_60_0 ctx)

def parallel_generatedRoot_897.w_62_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_897.w_63_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_18_4 ctx) (parallel_generatedRoot_897.w_62_0 ctx)

def parallel_generatedRoot_897.w_64_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_897.w_65_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_18_5 ctx) (parallel_generatedRoot_897.w_64_0 ctx)

def parallel_generatedRoot_897.w_66_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_897.w_67_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_897.w_68_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_66_0 ctx) (parallel_generatedRoot_897.w_67_0 ctx)

def parallel_generatedRoot_897.w_69_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_18_0 ctx) (parallel_generatedRoot_897.w_68_0 ctx)

def parallel_generatedRoot_897.w_70_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_897.w_71_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_897.w_72_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_70_0 ctx) (parallel_generatedRoot_897.w_71_0 ctx)

def parallel_generatedRoot_897.w_73_0  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_897.w_18_1 ctx) (parallel_generatedRoot_897.w_72_0 ctx)

def parallel_generatedRoot_897.outs  (ctx : parallel_generatedRoot_897Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_897.w_49_0 ctx), ((parallel_generatedRoot_897.w_61_0 ctx), ((parallel_generatedRoot_897.w_63_0 ctx), ((parallel_generatedRoot_897.w_65_0 ctx), ((parallel_generatedRoot_897.w_69_0 ctx), ((parallel_generatedRoot_897.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_897.Facts_0  (ctx : parallel_generatedRoot_897Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_897.w_1_0 ctx) (0) (parallel_generatedRoot_897.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_897.w_3_0 ctx) (0) (parallel_generatedRoot_897.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_897.w_5_0 ctx) (0) (parallel_generatedRoot_897.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_897.w_7_0 ctx) (0) (parallel_generatedRoot_897.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_897_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_897.w_9_0 ctx), ((parallel_generatedRoot_897.w_10_0 ctx), ((parallel_generatedRoot_897.w_11_0 ctx), ((parallel_generatedRoot_897.w_12_0 ctx), ((parallel_generatedRoot_897.w_13_0 ctx), ((parallel_generatedRoot_897.w_14_0 ctx), ((parallel_generatedRoot_897.w_15_0 ctx), ((parallel_generatedRoot_897.w_16_0 ctx), ((parallel_generatedRoot_897.w_17_0 ctx), ((parallel_generatedRoot_897.w_1_0 ctx), ((parallel_generatedRoot_897.w_3_0 ctx), ((parallel_generatedRoot_897.w_5_0 ctx), ((parallel_generatedRoot_897.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_897.w_0_0 ctx), ((parallel_generatedRoot_897.w_0_0 ctx), ((parallel_generatedRoot_897.w_0_0 ctx), ((parallel_generatedRoot_897.w_0_0 ctx), ((parallel_generatedRoot_897.w_0_0 ctx), ((parallel_generatedRoot_897.w_0_0 ctx), ((parallel_generatedRoot_897.w_2_0 ctx), ((parallel_generatedRoot_897.w_4_0 ctx), ((parallel_generatedRoot_897.w_6_0 ctx), ((parallel_generatedRoot_897.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_897.w_18_0 ctx), ((parallel_generatedRoot_897.w_18_1 ctx), ((parallel_generatedRoot_897.w_18_2 ctx), ((parallel_generatedRoot_897.w_18_3 ctx), ((parallel_generatedRoot_897.w_18_4 ctx), ((parallel_generatedRoot_897.w_18_5 ctx), ((parallel_generatedRoot_897.w_18_6 ctx), ((parallel_generatedRoot_897.w_18_7 ctx), ((parallel_generatedRoot_897.w_18_8 ctx), ((parallel_generatedRoot_897.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_897.w_9_0 ctx) (7) (parallel_generatedRoot_897.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_897.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_897.w_10_0 ctx)) (parallel_generatedRoot_897.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_897.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_897.w_10_0 ctx)) (parallel_generatedRoot_897.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_19_0 ctx) (parallel_generatedRoot_897.w_21_0 ctx) (parallel_generatedRoot_897.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_23_concat_1 ctx) (parallel_generatedRoot_897.w_22_0 ctx) (parallel_generatedRoot_897.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_897.w_11_0 ctx) (7) (parallel_generatedRoot_897.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_897.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_897.w_10_0 ctx)) (parallel_generatedRoot_897.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_897.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_897.w_10_0 ctx)) (parallel_generatedRoot_897.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_24_0 ctx) (parallel_generatedRoot_897.w_25_0 ctx) (parallel_generatedRoot_897.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_27_concat_1 ctx) (parallel_generatedRoot_897.w_26_0 ctx) (parallel_generatedRoot_897.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_897.w_23_0 ctx) (parallel_generatedRoot_897.w_27_0 ctx) (parallel_generatedRoot_897.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_897.w_12_0 ctx) (parallel_generatedRoot_897.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_897.w_13_0 ctx) (parallel_generatedRoot_897.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_897.w_14_0 ctx) (parallel_generatedRoot_897.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_29_0 ctx) (parallel_generatedRoot_897.w_30_0 ctx) (parallel_generatedRoot_897.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_32_concat_1 ctx) (parallel_generatedRoot_897.w_31_0 ctx) (parallel_generatedRoot_897.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_897.facts_0  {ctx : parallel_generatedRoot_897Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_897.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_897.w_1_0 ctx) (parallel_generatedRoot_897.w_3_0 ctx) (parallel_generatedRoot_897.w_5_0 ctx) (parallel_generatedRoot_897.w_7_0 ctx) (parallel_generatedRoot_897.w_9_0 ctx) (parallel_generatedRoot_897.w_10_0 ctx) (parallel_generatedRoot_897.w_11_0 ctx) (parallel_generatedRoot_897.w_12_0 ctx) (parallel_generatedRoot_897.w_13_0 ctx) (parallel_generatedRoot_897.w_14_0 ctx) (parallel_generatedRoot_897.w_15_0 ctx) (parallel_generatedRoot_897.w_16_0 ctx) (parallel_generatedRoot_897.w_17_0 ctx) (parallel_generatedRoot_897.w_0_0 ctx) (parallel_generatedRoot_897.w_2_0 ctx) (parallel_generatedRoot_897.w_4_0 ctx) (parallel_generatedRoot_897.w_6_0 ctx) (parallel_generatedRoot_897.w_8_0 ctx) (parallel_generatedRoot_897.w_18_0 ctx) (parallel_generatedRoot_897.w_18_1 ctx) (parallel_generatedRoot_897.w_18_2 ctx) (parallel_generatedRoot_897.w_18_3 ctx) (parallel_generatedRoot_897.w_18_4 ctx) (parallel_generatedRoot_897.w_18_5 ctx) (parallel_generatedRoot_897.w_18_6 ctx) (parallel_generatedRoot_897.w_18_7 ctx) (parallel_generatedRoot_897.w_18_8 ctx) (parallel_generatedRoot_897.w_18_9 ctx) (parallel_generatedRoot_897.w_19_0 ctx) (parallel_generatedRoot_897.w_20_0 ctx) (parallel_generatedRoot_897.w_21_0 ctx) (parallel_generatedRoot_897.w_22_0 ctx) (parallel_generatedRoot_897.w_23_0 ctx) (parallel_generatedRoot_897.w_23_concat_1 ctx) (parallel_generatedRoot_897.w_24_0 ctx) (parallel_generatedRoot_897.w_25_0 ctx) (parallel_generatedRoot_897.w_26_0 ctx) (parallel_generatedRoot_897.w_27_0 ctx) (parallel_generatedRoot_897.w_27_concat_1 ctx) (parallel_generatedRoot_897.w_28_0 ctx) (parallel_generatedRoot_897.w_29_0 ctx) (parallel_generatedRoot_897.w_30_0 ctx) (parallel_generatedRoot_897.w_31_0 ctx) (parallel_generatedRoot_897.w_32_0 ctx) (parallel_generatedRoot_897.w_32_concat_1 ctx)) :
    parallel_generatedRoot_897.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_897.Facts_1  (ctx : parallel_generatedRoot_897Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_897.w_15_0 ctx) (parallel_generatedRoot_897.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_897.w_16_0 ctx) (parallel_generatedRoot_897.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_897.w_17_0 ctx) (parallel_generatedRoot_897.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_33_0 ctx) (parallel_generatedRoot_897.w_34_0 ctx) (parallel_generatedRoot_897.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_36_concat_1 ctx) (parallel_generatedRoot_897.w_35_0 ctx) (parallel_generatedRoot_897.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_897.w_32_0 ctx) (parallel_generatedRoot_897.w_36_0 ctx) (parallel_generatedRoot_897.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_18_6 ctx) (parallel_generatedRoot_897.w_18_8 ctx) (parallel_generatedRoot_897.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_897.w_38_0 ctx) (parallel_generatedRoot_897.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_897.w_39_0 ctx) (parallel_generatedRoot_897.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_897.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_897.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_42_0 ctx) (parallel_generatedRoot_897.w_43_0 ctx) (parallel_generatedRoot_897.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_897.w_44_0 ctx) (parallel_generatedRoot_897.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_897.w_45_0 ctx) (parallel_generatedRoot_897.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_897.facts_1  {ctx : parallel_generatedRoot_897Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_897.constraints_1 backend (parallel_generatedRoot_897.w_15_0 ctx) (parallel_generatedRoot_897.w_16_0 ctx) (parallel_generatedRoot_897.w_17_0 ctx) (parallel_generatedRoot_897.w_18_6 ctx) (parallel_generatedRoot_897.w_18_8 ctx) (parallel_generatedRoot_897.w_32_0 ctx) (parallel_generatedRoot_897.w_33_0 ctx) (parallel_generatedRoot_897.w_34_0 ctx) (parallel_generatedRoot_897.w_35_0 ctx) (parallel_generatedRoot_897.w_36_0 ctx) (parallel_generatedRoot_897.w_36_concat_1 ctx) (parallel_generatedRoot_897.w_37_0 ctx) (parallel_generatedRoot_897.w_38_0 ctx) (parallel_generatedRoot_897.w_39_0 ctx) (parallel_generatedRoot_897.w_40_0 ctx) (parallel_generatedRoot_897.w_41_0 ctx) (parallel_generatedRoot_897.w_42_0 ctx) (parallel_generatedRoot_897.w_43_0 ctx) (parallel_generatedRoot_897.w_44_0 ctx) (parallel_generatedRoot_897.w_45_0 ctx) (parallel_generatedRoot_897.w_46_0 ctx)) :
    parallel_generatedRoot_897.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_897.Facts_2  (ctx : parallel_generatedRoot_897Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_897.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_18_7 ctx) (parallel_generatedRoot_897.w_18_9 ctx) (parallel_generatedRoot_897.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_897.w_50_0 ctx) (parallel_generatedRoot_897.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_897.w_51_0 ctx) (parallel_generatedRoot_897.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_897.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_897.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_897.w_54_0 ctx) (parallel_generatedRoot_897.w_55_0 ctx) (parallel_generatedRoot_897.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_897.w_56_0 ctx) (parallel_generatedRoot_897.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_897.w_57_0 ctx) (parallel_generatedRoot_897.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_897.w_60_0 ctx)

theorem parallel_generatedRoot_897.facts_2  {ctx : parallel_generatedRoot_897Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_897.constraints_2 backend (parallel_generatedRoot_897.w_18_7 ctx) (parallel_generatedRoot_897.w_18_9 ctx) (parallel_generatedRoot_897.w_47_0 ctx) (parallel_generatedRoot_897.w_48_0 ctx) (parallel_generatedRoot_897.w_50_0 ctx) (parallel_generatedRoot_897.w_51_0 ctx) (parallel_generatedRoot_897.w_52_0 ctx) (parallel_generatedRoot_897.w_53_0 ctx) (parallel_generatedRoot_897.w_54_0 ctx) (parallel_generatedRoot_897.w_55_0 ctx) (parallel_generatedRoot_897.w_56_0 ctx) (parallel_generatedRoot_897.w_57_0 ctx) (parallel_generatedRoot_897.w_58_0 ctx) (parallel_generatedRoot_897.w_59_0 ctx) (parallel_generatedRoot_897.w_60_0 ctx)) :
    parallel_generatedRoot_897.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_897.Facts_3  (ctx : parallel_generatedRoot_897Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_897.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_897.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_897.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_897.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_897.facts_3  {ctx : parallel_generatedRoot_897Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_897.constraints_3 (parallel_generatedRoot_897.w_41_0 ctx) (parallel_generatedRoot_897.w_47_0 ctx) (parallel_generatedRoot_897.w_59_0 ctx) (parallel_generatedRoot_897.w_62_0 ctx) (parallel_generatedRoot_897.w_64_0 ctx) (parallel_generatedRoot_897.w_66_0 ctx) (parallel_generatedRoot_897.w_67_0 ctx)) :
    parallel_generatedRoot_897.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_897.Facts_4  (ctx : parallel_generatedRoot_897Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_897.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_897.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_897.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_897.outs ctx

theorem parallel_generatedRoot_897.facts_4  {ctx : parallel_generatedRoot_897Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_897.constraints_4 (parallel_generatedRoot_897.w_49_0 ctx) (parallel_generatedRoot_897.w_53_0 ctx) (parallel_generatedRoot_897.w_59_0 ctx) (parallel_generatedRoot_897.w_61_0 ctx) (parallel_generatedRoot_897.w_63_0 ctx) (parallel_generatedRoot_897.w_65_0 ctx) (parallel_generatedRoot_897.w_69_0 ctx) (parallel_generatedRoot_897.w_70_0 ctx) (parallel_generatedRoot_897.w_71_0 ctx) (parallel_generatedRoot_897.w_73_0 ctx) outputs) :
    parallel_generatedRoot_897.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_897.Facts  (ctx : parallel_generatedRoot_897Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_897.Facts_0 ctx
  c1 : parallel_generatedRoot_897.Facts_1 ctx
  c2 : parallel_generatedRoot_897.Facts_2 ctx
  c3 : parallel_generatedRoot_897.Facts_3 ctx
  c4 : parallel_generatedRoot_897.Facts_4 ctx outputs

theorem parallel_generatedRoot_897.facts  (ctx : parallel_generatedRoot_897Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_897.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_897.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_897.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_897.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_897.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_897.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_897.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_929.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (sequential_generatedRoot_929.InputType)
  witness : sequential_generatedRoot_929.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_929Ctx" => (sequential_generatedRoot_929.Ctx backend hashModel params i_0)

def sequential_generatedRoot_929.w_0_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.inputs.1

def sequential_generatedRoot_929.w_31_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.inputs.2.1

def sequential_generatedRoot_929.w_35_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_929.w_39_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_929.w_43_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_929.w_47_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_929.w_2_0  (ctx : sequential_generatedRoot_929Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_929.w_8_0  (ctx : sequential_generatedRoot_929Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_929.w_12_0  (ctx : sequential_generatedRoot_929Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_929.w_16_0  (ctx : sequential_generatedRoot_929Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_929.w_20_0  (ctx : sequential_generatedRoot_929Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_929.w_24_0  (ctx : sequential_generatedRoot_929Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_929.w_1_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_929.w_3_0  (ctx : sequential_generatedRoot_929Ctx) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_929.w_4_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_929.w_5_0  (ctx : sequential_generatedRoot_929Ctx) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_929.w_6_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_929.w_7_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_4_0 ctx) (sequential_generatedRoot_929.w_6_0 ctx)

def sequential_generatedRoot_929.w_9_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_929.w_10_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_929.w_11_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_9_0 ctx) (sequential_generatedRoot_929.w_10_0 ctx)

def sequential_generatedRoot_929.w_13_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_929.w_14_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_929.w_15_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_13_0 ctx) (sequential_generatedRoot_929.w_14_0 ctx)

def sequential_generatedRoot_929.w_17_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_929.w_18_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_929.w_19_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_17_0 ctx) (sequential_generatedRoot_929.w_18_0 ctx)

def sequential_generatedRoot_929.w_21_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_929.w_22_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def sequential_generatedRoot_929.w_23_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_21_0 ctx) (sequential_generatedRoot_929.w_22_0 ctx)

def sequential_generatedRoot_929.w_25_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_929.w_26_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def sequential_generatedRoot_929.w_27_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_25_0 ctx) (sequential_generatedRoot_929.w_26_0 ctx)

def sequential_generatedRoot_929.w_28_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_28_0

def sequential_generatedRoot_929.w_28_1  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_28_1

def sequential_generatedRoot_929.w_28_2  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_28_2

def sequential_generatedRoot_929.w_28_3  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_28_3

def sequential_generatedRoot_929.w_28_4  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_28_4

def sequential_generatedRoot_929.w_28_5  (ctx : sequential_generatedRoot_929Ctx) : mxxMat0 :=
  ctx.witness.w_28_5

def sequential_generatedRoot_929.w_29_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_929.w_30_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_0_0 ctx) (sequential_generatedRoot_929.w_29_0 ctx)

def sequential_generatedRoot_929.w_32_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_929.w_33_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_929.w_34_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_31_0 ctx) (sequential_generatedRoot_929.w_33_0 ctx)

def sequential_generatedRoot_929.w_36_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_929.w_37_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.witness.w_37_0

def sequential_generatedRoot_929.w_38_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_35_0 ctx) (sequential_generatedRoot_929.w_37_0 ctx)

def sequential_generatedRoot_929.w_40_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_929.w_41_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.witness.w_41_0

def sequential_generatedRoot_929.w_42_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_39_0 ctx) (sequential_generatedRoot_929.w_41_0 ctx)

def sequential_generatedRoot_929.w_44_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_929.w_45_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.witness.w_45_0

def sequential_generatedRoot_929.w_46_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_43_0 ctx) (sequential_generatedRoot_929.w_45_0 ctx)

def sequential_generatedRoot_929.w_48_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_929.w_49_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  ctx.witness.w_49_0

def sequential_generatedRoot_929.w_50_0  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_929.w_47_0 ctx) (sequential_generatedRoot_929.w_49_0 ctx)

def sequential_generatedRoot_929.outs  (ctx : sequential_generatedRoot_929Ctx) : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit :=
  ((sequential_generatedRoot_929.w_30_0 ctx), ((sequential_generatedRoot_929.w_34_0 ctx), ((sequential_generatedRoot_929.w_38_0 ctx), ((sequential_generatedRoot_929.w_42_0 ctx), ((sequential_generatedRoot_929.w_46_0 ctx), ((sequential_generatedRoot_929.w_50_0 ctx), ()))))))

structure sequential_generatedRoot_929.Facts_0  (ctx : sequential_generatedRoot_929Ctx) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 900
  h3_0 : 0 ≤ (sequential_generatedRoot_929.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_929.w_3_0 ctx) < 7200
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_2_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_929.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_929.w_5_0 ctx) < 7200
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_2_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_929.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_929.w_3_0 ctx) < 7200
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_8_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_929.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_929.w_5_0 ctx) < 7200
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_8_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_929.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_929.w_3_0 ctx) < 7200
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_12_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_929.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_929.w_5_0 ctx) < 7200
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_12_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_929.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_929.w_3_0 ctx) < 7200
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_16_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_929.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_929.w_5_0 ctx) < 7200
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_16_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_18_0 ctx)
  h3_0_8 : 0 ≤ (sequential_generatedRoot_929.w_3_0 ctx)
  h3_0_9 : (sequential_generatedRoot_929.w_3_0 ctx) < 7200
  h21_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_20_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_21_0 ctx)
  h5_0_8 : 0 ≤ (sequential_generatedRoot_929.w_5_0 ctx)
  h5_0_9 : (sequential_generatedRoot_929.w_5_0 ctx) < 7200
  h22_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_20_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_22_0 ctx)

theorem sequential_generatedRoot_929.facts_0  {ctx : sequential_generatedRoot_929Ctx} {outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit}
    (h : sequential_generatedRoot_929.constraints_0 i_0 (sequential_generatedRoot_929.w_2_0 ctx) (sequential_generatedRoot_929.w_8_0 ctx) (sequential_generatedRoot_929.w_12_0 ctx) (sequential_generatedRoot_929.w_16_0 ctx) (sequential_generatedRoot_929.w_20_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_4_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_6_0 ctx) (sequential_generatedRoot_929.w_9_0 ctx) (sequential_generatedRoot_929.w_10_0 ctx) (sequential_generatedRoot_929.w_13_0 ctx) (sequential_generatedRoot_929.w_14_0 ctx) (sequential_generatedRoot_929.w_17_0 ctx) (sequential_generatedRoot_929.w_18_0 ctx) (sequential_generatedRoot_929.w_21_0 ctx) (sequential_generatedRoot_929.w_22_0 ctx)) :
    sequential_generatedRoot_929.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_929.Facts_1  (ctx : sequential_generatedRoot_929Ctx) (outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) : Prop where
  h3_0_10 : 0 ≤ (sequential_generatedRoot_929.w_3_0 ctx)
  h3_0_11 : (sequential_generatedRoot_929.w_3_0 ctx) < 7200
  h25_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_24_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_25_0 ctx)
  h5_0_10 : 0 ≤ (sequential_generatedRoot_929.w_5_0 ctx)
  h5_0_11 : (sequential_generatedRoot_929.w_5_0 ctx) < 7200
  h26_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_929.w_24_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_26_0 ctx)
  t38 : 0 ≤ (6)
  h28_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_88 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((sequential_generatedRoot_929.w_2_0 ctx), ((sequential_generatedRoot_929.w_8_0 ctx), ((sequential_generatedRoot_929.w_12_0 ctx), ((sequential_generatedRoot_929.w_16_0 ctx), ((sequential_generatedRoot_929.w_20_0 ctx), ((sequential_generatedRoot_929.w_24_0 ctx), ())))))))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_929.w_7_0 ctx), ((sequential_generatedRoot_929.w_11_0 ctx), ((sequential_generatedRoot_929.w_15_0 ctx), ((sequential_generatedRoot_929.w_19_0 ctx), ((sequential_generatedRoot_929.w_23_0 ctx), ((sequential_generatedRoot_929.w_27_0 ctx), ())))))) ((sequential_generatedRoot_929.w_28_0 ctx), ((sequential_generatedRoot_929.w_28_1 ctx), ((sequential_generatedRoot_929.w_28_2 ctx), ((sequential_generatedRoot_929.w_28_3 ctx), ((sequential_generatedRoot_929.w_28_4 ctx), ((sequential_generatedRoot_929.w_28_5 ctx), ()))))))
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_929.w_1_0 ctx) (sequential_generatedRoot_929.w_28_0 ctx) (sequential_generatedRoot_929.w_29_0 ctx)
  t41 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t42 : ((1 + (Int.ofNat i_0))) < 900
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_929.w_32_0 ctx) (sequential_generatedRoot_929.w_28_1 ctx) (sequential_generatedRoot_929.w_33_0 ctx)
  t44 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t45 : ((1 + (Int.ofNat i_0))) < 900
  h37_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_929.w_36_0 ctx) (sequential_generatedRoot_929.w_28_2 ctx) (sequential_generatedRoot_929.w_37_0 ctx)
  t47 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t48 : ((1 + (Int.ofNat i_0))) < 900
  h41_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_929.w_40_0 ctx) (sequential_generatedRoot_929.w_28_3 ctx) (sequential_generatedRoot_929.w_41_0 ctx)
  t50 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t51 : ((1 + (Int.ofNat i_0))) < 900
  h45_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_929.w_44_0 ctx) (sequential_generatedRoot_929.w_28_4 ctx) (sequential_generatedRoot_929.w_45_0 ctx)
  t53 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t54 : ((1 + (Int.ofNat i_0))) < 900
  h49_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_929.w_48_0 ctx) (sequential_generatedRoot_929.w_28_5 ctx) (sequential_generatedRoot_929.w_49_0 ctx)
  hout : outputs = sequential_generatedRoot_929.outs ctx

theorem sequential_generatedRoot_929.facts_1  {ctx : sequential_generatedRoot_929Ctx} {outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit}
    (h : sequential_generatedRoot_929.constraints_1 backend hashModel params i_0 (sequential_generatedRoot_929.w_2_0 ctx) (sequential_generatedRoot_929.w_8_0 ctx) (sequential_generatedRoot_929.w_12_0 ctx) (sequential_generatedRoot_929.w_16_0 ctx) (sequential_generatedRoot_929.w_20_0 ctx) (sequential_generatedRoot_929.w_24_0 ctx) (sequential_generatedRoot_929.w_1_0 ctx) (sequential_generatedRoot_929.w_3_0 ctx) (sequential_generatedRoot_929.w_5_0 ctx) (sequential_generatedRoot_929.w_7_0 ctx) (sequential_generatedRoot_929.w_11_0 ctx) (sequential_generatedRoot_929.w_15_0 ctx) (sequential_generatedRoot_929.w_19_0 ctx) (sequential_generatedRoot_929.w_23_0 ctx) (sequential_generatedRoot_929.w_25_0 ctx) (sequential_generatedRoot_929.w_26_0 ctx) (sequential_generatedRoot_929.w_27_0 ctx) (sequential_generatedRoot_929.w_28_0 ctx) (sequential_generatedRoot_929.w_28_1 ctx) (sequential_generatedRoot_929.w_28_2 ctx) (sequential_generatedRoot_929.w_28_3 ctx) (sequential_generatedRoot_929.w_28_4 ctx) (sequential_generatedRoot_929.w_28_5 ctx) (sequential_generatedRoot_929.w_29_0 ctx) (sequential_generatedRoot_929.w_30_0 ctx) (sequential_generatedRoot_929.w_32_0 ctx) (sequential_generatedRoot_929.w_33_0 ctx) (sequential_generatedRoot_929.w_34_0 ctx) (sequential_generatedRoot_929.w_36_0 ctx) (sequential_generatedRoot_929.w_37_0 ctx) (sequential_generatedRoot_929.w_38_0 ctx) (sequential_generatedRoot_929.w_40_0 ctx) (sequential_generatedRoot_929.w_41_0 ctx) (sequential_generatedRoot_929.w_42_0 ctx) (sequential_generatedRoot_929.w_44_0 ctx) (sequential_generatedRoot_929.w_45_0 ctx) (sequential_generatedRoot_929.w_46_0 ctx) (sequential_generatedRoot_929.w_48_0 ctx) (sequential_generatedRoot_929.w_49_0 ctx) (sequential_generatedRoot_929.w_50_0 ctx) outputs) :
    sequential_generatedRoot_929.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_929.Facts  (ctx : sequential_generatedRoot_929Ctx) (outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) : Prop where
  c0 : sequential_generatedRoot_929.Facts_0 ctx
  c1 : sequential_generatedRoot_929.Facts_1 ctx outputs

theorem sequential_generatedRoot_929.facts  (ctx : sequential_generatedRoot_929Ctx) {outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit}
    (h : sequential_generatedRoot_929.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_929.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_929.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_929.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_946.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit
  witness : parallel_generatedRoot_946.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_946Ctx" => (parallel_generatedRoot_946.Ctx backend hashModel params i_0)

def parallel_generatedRoot_946.w_0_0  (ctx : parallel_generatedRoot_946Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_946.w_1_0  (ctx : parallel_generatedRoot_946Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_946.w_2_0  (ctx : parallel_generatedRoot_946Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_946.w_3_0  (ctx : parallel_generatedRoot_946Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_946.w_4_0  (ctx : parallel_generatedRoot_946Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_946.w_5_0  (ctx : parallel_generatedRoot_946Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_946.w_6_0  (ctx : parallel_generatedRoot_946Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_946.w_6_1  (ctx : parallel_generatedRoot_946Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_946.w_6_2  (ctx : parallel_generatedRoot_946Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_946.w_6_3  (ctx : parallel_generatedRoot_946Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_946.w_6_4  (ctx : parallel_generatedRoot_946Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_946.w_6_5  (ctx : parallel_generatedRoot_946Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_946.w_7_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_946.w_8_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_946.w_9_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_946.w_7_0 ctx) (parallel_generatedRoot_946.w_8_0 ctx)

def parallel_generatedRoot_946.w_10_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_946.w_11_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_946.w_12_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_946.w_10_0 ctx) (parallel_generatedRoot_946.w_11_0 ctx)

def parallel_generatedRoot_946.w_13_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_946.w_14_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_946.w_15_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_946.w_13_0 ctx) (parallel_generatedRoot_946.w_14_0 ctx)

def parallel_generatedRoot_946.w_16_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_946.w_17_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_946.w_18_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_946.w_16_0 ctx) (parallel_generatedRoot_946.w_17_0 ctx)

def parallel_generatedRoot_946.w_19_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_946.w_20_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_946.w_21_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_946.w_19_0 ctx) (parallel_generatedRoot_946.w_20_0 ctx)

def parallel_generatedRoot_946.w_22_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_946.w_23_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_946.w_24_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_946.w_22_0 ctx) (parallel_generatedRoot_946.w_23_0 ctx)

def parallel_generatedRoot_946.w_25_0  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_946.w_25_1  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_946.w_25_2  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_946.w_25_3  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_946.w_25_4  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_946.w_25_5  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_946.outs  (ctx : parallel_generatedRoot_946Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_946.w_25_0 ctx), ((parallel_generatedRoot_946.w_25_1 ctx), ((parallel_generatedRoot_946.w_25_2 ctx), ((parallel_generatedRoot_946.w_25_3 ctx), ((parallel_generatedRoot_946.w_25_4 ctx), ((parallel_generatedRoot_946.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_946.Facts_0  (ctx : parallel_generatedRoot_946Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_946_6 backend hashModel params i_0 i ((parallel_generatedRoot_946.w_0_0 ctx), ((parallel_generatedRoot_946.w_1_0 ctx), ((parallel_generatedRoot_946.w_2_0 ctx), ((parallel_generatedRoot_946.w_3_0 ctx), ((parallel_generatedRoot_946.w_4_0 ctx), ((parallel_generatedRoot_946.w_5_0 ctx), ())))))) (((parallel_generatedRoot_946.w_6_0 ctx) i), (((parallel_generatedRoot_946.w_6_1 ctx) i), (((parallel_generatedRoot_946.w_6_2 ctx) i), (((parallel_generatedRoot_946.w_6_3 ctx) i), (((parallel_generatedRoot_946.w_6_4 ctx) i), (((parallel_generatedRoot_946.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_0 ctx) (0) (parallel_generatedRoot_946.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_0 ctx) (1) (parallel_generatedRoot_946.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_1 ctx) (0) (parallel_generatedRoot_946.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_1 ctx) (1) (parallel_generatedRoot_946.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_2 ctx) (0) (parallel_generatedRoot_946.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_2 ctx) (1) (parallel_generatedRoot_946.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_3 ctx) (0) (parallel_generatedRoot_946.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_3 ctx) (1) (parallel_generatedRoot_946.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_4 ctx) (0) (parallel_generatedRoot_946.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_4 ctx) (1) (parallel_generatedRoot_946.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_5 ctx) (0) (parallel_generatedRoot_946.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_946.w_6_5 ctx) (1) (parallel_generatedRoot_946.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_946_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_946.w_6_0 ctx), ((parallel_generatedRoot_946.w_6_1 ctx), ((parallel_generatedRoot_946.w_6_2 ctx), ((parallel_generatedRoot_946.w_6_3 ctx), ((parallel_generatedRoot_946.w_6_4 ctx), ((parallel_generatedRoot_946.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_946.w_9_0 ctx), ((parallel_generatedRoot_946.w_12_0 ctx), ((parallel_generatedRoot_946.w_15_0 ctx), ((parallel_generatedRoot_946.w_18_0 ctx), ((parallel_generatedRoot_946.w_21_0 ctx), ((parallel_generatedRoot_946.w_24_0 ctx), ())))))) ((parallel_generatedRoot_946.w_25_0 ctx), ((parallel_generatedRoot_946.w_25_1 ctx), ((parallel_generatedRoot_946.w_25_2 ctx), ((parallel_generatedRoot_946.w_25_3 ctx), ((parallel_generatedRoot_946.w_25_4 ctx), ((parallel_generatedRoot_946.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_946.outs ctx

theorem parallel_generatedRoot_946.facts_0  {ctx : parallel_generatedRoot_946Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_946.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_946.w_0_0 ctx) (parallel_generatedRoot_946.w_1_0 ctx) (parallel_generatedRoot_946.w_2_0 ctx) (parallel_generatedRoot_946.w_3_0 ctx) (parallel_generatedRoot_946.w_4_0 ctx) (parallel_generatedRoot_946.w_5_0 ctx) (parallel_generatedRoot_946.w_6_0 ctx) (parallel_generatedRoot_946.w_6_1 ctx) (parallel_generatedRoot_946.w_6_2 ctx) (parallel_generatedRoot_946.w_6_3 ctx) (parallel_generatedRoot_946.w_6_4 ctx) (parallel_generatedRoot_946.w_6_5 ctx) (parallel_generatedRoot_946.w_7_0 ctx) (parallel_generatedRoot_946.w_8_0 ctx) (parallel_generatedRoot_946.w_9_0 ctx) (parallel_generatedRoot_946.w_10_0 ctx) (parallel_generatedRoot_946.w_11_0 ctx) (parallel_generatedRoot_946.w_12_0 ctx) (parallel_generatedRoot_946.w_13_0 ctx) (parallel_generatedRoot_946.w_14_0 ctx) (parallel_generatedRoot_946.w_15_0 ctx) (parallel_generatedRoot_946.w_16_0 ctx) (parallel_generatedRoot_946.w_17_0 ctx) (parallel_generatedRoot_946.w_18_0 ctx) (parallel_generatedRoot_946.w_19_0 ctx) (parallel_generatedRoot_946.w_20_0 ctx) (parallel_generatedRoot_946.w_21_0 ctx) (parallel_generatedRoot_946.w_22_0 ctx) (parallel_generatedRoot_946.w_23_0 ctx) (parallel_generatedRoot_946.w_24_0 ctx) (parallel_generatedRoot_946.w_25_0 ctx) (parallel_generatedRoot_946.w_25_1 ctx) (parallel_generatedRoot_946.w_25_2 ctx) (parallel_generatedRoot_946.w_25_3 ctx) (parallel_generatedRoot_946.w_25_4 ctx) (parallel_generatedRoot_946.w_25_5 ctx) outputs) :
    parallel_generatedRoot_946.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_946.Facts  (ctx : parallel_generatedRoot_946Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_946.Facts_0 ctx outputs

theorem parallel_generatedRoot_946.facts  (ctx : parallel_generatedRoot_946Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_946.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_946.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_946.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_950.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit
  witness : parallel_generatedRoot_950.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_950Ctx" => (parallel_generatedRoot_950.Ctx backend hashModel params i_0)

def parallel_generatedRoot_950.w_0_0  (ctx : parallel_generatedRoot_950Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_950.w_1_0  (ctx : parallel_generatedRoot_950Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_950.w_2_0  (ctx : parallel_generatedRoot_950Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_950.w_3_0  (ctx : parallel_generatedRoot_950Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_950.w_4_0  (ctx : parallel_generatedRoot_950Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_950.w_5_0  (ctx : parallel_generatedRoot_950Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_950.w_6_0  (ctx : parallel_generatedRoot_950Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_950.w_6_1  (ctx : parallel_generatedRoot_950Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_950.w_6_2  (ctx : parallel_generatedRoot_950Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_950.w_6_3  (ctx : parallel_generatedRoot_950Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_950.w_6_4  (ctx : parallel_generatedRoot_950Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_950.w_6_5  (ctx : parallel_generatedRoot_950Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_950.w_7_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_950.w_8_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_950.w_9_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_950.w_7_0 ctx) (parallel_generatedRoot_950.w_8_0 ctx)

def parallel_generatedRoot_950.w_10_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_950.w_11_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_950.w_12_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_950.w_10_0 ctx) (parallel_generatedRoot_950.w_11_0 ctx)

def parallel_generatedRoot_950.w_13_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_950.w_14_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_950.w_15_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_950.w_13_0 ctx) (parallel_generatedRoot_950.w_14_0 ctx)

def parallel_generatedRoot_950.w_16_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_950.w_17_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_950.w_18_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_950.w_16_0 ctx) (parallel_generatedRoot_950.w_17_0 ctx)

def parallel_generatedRoot_950.w_19_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_950.w_20_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_950.w_21_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_950.w_19_0 ctx) (parallel_generatedRoot_950.w_20_0 ctx)

def parallel_generatedRoot_950.w_22_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_950.w_23_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_950.w_24_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_950.w_22_0 ctx) (parallel_generatedRoot_950.w_23_0 ctx)

def parallel_generatedRoot_950.w_25_0  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_950.w_25_1  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_950.w_25_2  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_950.w_25_3  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_950.w_25_4  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_950.w_25_5  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_950.outs  (ctx : parallel_generatedRoot_950Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_950.w_25_0 ctx), ((parallel_generatedRoot_950.w_25_1 ctx), ((parallel_generatedRoot_950.w_25_2 ctx), ((parallel_generatedRoot_950.w_25_3 ctx), ((parallel_generatedRoot_950.w_25_4 ctx), ((parallel_generatedRoot_950.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_950.Facts_0  (ctx : parallel_generatedRoot_950Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_950_6 backend hashModel params i_0 i ((parallel_generatedRoot_950.w_0_0 ctx), ((parallel_generatedRoot_950.w_1_0 ctx), ((parallel_generatedRoot_950.w_2_0 ctx), ((parallel_generatedRoot_950.w_3_0 ctx), ((parallel_generatedRoot_950.w_4_0 ctx), ((parallel_generatedRoot_950.w_5_0 ctx), ())))))) (((parallel_generatedRoot_950.w_6_0 ctx) i), (((parallel_generatedRoot_950.w_6_1 ctx) i), (((parallel_generatedRoot_950.w_6_2 ctx) i), (((parallel_generatedRoot_950.w_6_3 ctx) i), (((parallel_generatedRoot_950.w_6_4 ctx) i), (((parallel_generatedRoot_950.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_0 ctx) (0) (parallel_generatedRoot_950.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_0 ctx) (1) (parallel_generatedRoot_950.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_1 ctx) (0) (parallel_generatedRoot_950.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_1 ctx) (1) (parallel_generatedRoot_950.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_2 ctx) (0) (parallel_generatedRoot_950.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_2 ctx) (1) (parallel_generatedRoot_950.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_3 ctx) (0) (parallel_generatedRoot_950.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_3 ctx) (1) (parallel_generatedRoot_950.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_4 ctx) (0) (parallel_generatedRoot_950.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_4 ctx) (1) (parallel_generatedRoot_950.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_5 ctx) (0) (parallel_generatedRoot_950.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_950.w_6_5 ctx) (1) (parallel_generatedRoot_950.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_950_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_950.w_6_0 ctx), ((parallel_generatedRoot_950.w_6_1 ctx), ((parallel_generatedRoot_950.w_6_2 ctx), ((parallel_generatedRoot_950.w_6_3 ctx), ((parallel_generatedRoot_950.w_6_4 ctx), ((parallel_generatedRoot_950.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_950.w_9_0 ctx), ((parallel_generatedRoot_950.w_12_0 ctx), ((parallel_generatedRoot_950.w_15_0 ctx), ((parallel_generatedRoot_950.w_18_0 ctx), ((parallel_generatedRoot_950.w_21_0 ctx), ((parallel_generatedRoot_950.w_24_0 ctx), ())))))) ((parallel_generatedRoot_950.w_25_0 ctx), ((parallel_generatedRoot_950.w_25_1 ctx), ((parallel_generatedRoot_950.w_25_2 ctx), ((parallel_generatedRoot_950.w_25_3 ctx), ((parallel_generatedRoot_950.w_25_4 ctx), ((parallel_generatedRoot_950.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_950.outs ctx

theorem parallel_generatedRoot_950.facts_0  {ctx : parallel_generatedRoot_950Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_950.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_950.w_0_0 ctx) (parallel_generatedRoot_950.w_1_0 ctx) (parallel_generatedRoot_950.w_2_0 ctx) (parallel_generatedRoot_950.w_3_0 ctx) (parallel_generatedRoot_950.w_4_0 ctx) (parallel_generatedRoot_950.w_5_0 ctx) (parallel_generatedRoot_950.w_6_0 ctx) (parallel_generatedRoot_950.w_6_1 ctx) (parallel_generatedRoot_950.w_6_2 ctx) (parallel_generatedRoot_950.w_6_3 ctx) (parallel_generatedRoot_950.w_6_4 ctx) (parallel_generatedRoot_950.w_6_5 ctx) (parallel_generatedRoot_950.w_7_0 ctx) (parallel_generatedRoot_950.w_8_0 ctx) (parallel_generatedRoot_950.w_9_0 ctx) (parallel_generatedRoot_950.w_10_0 ctx) (parallel_generatedRoot_950.w_11_0 ctx) (parallel_generatedRoot_950.w_12_0 ctx) (parallel_generatedRoot_950.w_13_0 ctx) (parallel_generatedRoot_950.w_14_0 ctx) (parallel_generatedRoot_950.w_15_0 ctx) (parallel_generatedRoot_950.w_16_0 ctx) (parallel_generatedRoot_950.w_17_0 ctx) (parallel_generatedRoot_950.w_18_0 ctx) (parallel_generatedRoot_950.w_19_0 ctx) (parallel_generatedRoot_950.w_20_0 ctx) (parallel_generatedRoot_950.w_21_0 ctx) (parallel_generatedRoot_950.w_22_0 ctx) (parallel_generatedRoot_950.w_23_0 ctx) (parallel_generatedRoot_950.w_24_0 ctx) (parallel_generatedRoot_950.w_25_0 ctx) (parallel_generatedRoot_950.w_25_1 ctx) (parallel_generatedRoot_950.w_25_2 ctx) (parallel_generatedRoot_950.w_25_3 ctx) (parallel_generatedRoot_950.w_25_4 ctx) (parallel_generatedRoot_950.w_25_5 ctx) outputs) :
    parallel_generatedRoot_950.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_950.Facts  (ctx : parallel_generatedRoot_950Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_950.Facts_0 ctx outputs

theorem parallel_generatedRoot_950.facts  (ctx : parallel_generatedRoot_950Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_950.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_950.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_950.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_951.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_951.InputType)
  witness : parallel_generatedRoot_951.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_951Ctx" => (parallel_generatedRoot_951.Ctx backend hashModel params i_0)

def parallel_generatedRoot_951.w_1_0  (ctx : parallel_generatedRoot_951Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_951.w_3_0  (ctx : parallel_generatedRoot_951Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_951.w_5_0  (ctx : parallel_generatedRoot_951Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_951.w_7_0  (ctx : parallel_generatedRoot_951Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_951.w_9_0  (ctx : parallel_generatedRoot_951Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_951.w_10_0  (ctx : parallel_generatedRoot_951Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_951.w_11_0  (ctx : parallel_generatedRoot_951Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_951.w_12_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_951.w_13_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_951.w_14_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_951.w_15_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_951.w_16_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_951.w_17_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_951.w_0_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_951.w_2_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_951.w_4_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_951.w_6_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_951.w_8_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_951.w_18_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_951.w_18_1  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_951.w_18_2  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_951.w_18_3  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_951.w_18_4  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_951.w_18_5  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_951.w_18_6  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_951.w_18_7  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_951.w_18_8  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_951.w_18_9  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_951.w_19_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_951.w_20_0  (ctx : parallel_generatedRoot_951Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_951.w_21_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_951.w_22_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_951.w_23_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_951.w_23_concat_1  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_951.w_24_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_951.w_25_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_951.w_26_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_951.w_27_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_951.w_27_concat_1  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_951.w_28_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_951.w_29_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_951.w_30_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_951.w_31_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_951.w_32_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_951.w_32_concat_1  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_951.w_33_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_951.w_34_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_951.w_35_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_951.w_36_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_951.w_36_concat_1  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_951.w_37_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_951.w_38_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_951.w_39_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_951.w_40_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_951.w_41_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_951.w_37_0 ctx) (parallel_generatedRoot_951.w_40_0 ctx)

def parallel_generatedRoot_951.w_42_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_951.w_43_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_951.w_44_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_951.w_45_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_951.w_46_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_951.w_47_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_951.w_28_0 ctx) (parallel_generatedRoot_951.w_46_0 ctx)

def parallel_generatedRoot_951.w_48_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_951.w_49_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_18_2 ctx) (parallel_generatedRoot_951.w_48_0 ctx)

def parallel_generatedRoot_951.w_50_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_951.w_51_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_951.w_52_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_951.w_53_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_951.w_37_0 ctx) (parallel_generatedRoot_951.w_52_0 ctx)

def parallel_generatedRoot_951.w_54_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_951.w_55_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_951.w_56_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_951.w_57_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_951.w_58_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_951.w_59_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_951.w_28_0 ctx) (parallel_generatedRoot_951.w_58_0 ctx)

def parallel_generatedRoot_951.w_60_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_951.w_61_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_18_3 ctx) (parallel_generatedRoot_951.w_60_0 ctx)

def parallel_generatedRoot_951.w_62_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_951.w_63_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_18_4 ctx) (parallel_generatedRoot_951.w_62_0 ctx)

def parallel_generatedRoot_951.w_64_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_951.w_65_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_18_5 ctx) (parallel_generatedRoot_951.w_64_0 ctx)

def parallel_generatedRoot_951.w_66_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_951.w_67_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_951.w_68_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_66_0 ctx) (parallel_generatedRoot_951.w_67_0 ctx)

def parallel_generatedRoot_951.w_69_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_18_0 ctx) (parallel_generatedRoot_951.w_68_0 ctx)

def parallel_generatedRoot_951.w_70_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_951.w_71_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_951.w_72_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_70_0 ctx) (parallel_generatedRoot_951.w_71_0 ctx)

def parallel_generatedRoot_951.w_73_0  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_951.w_18_1 ctx) (parallel_generatedRoot_951.w_72_0 ctx)

def parallel_generatedRoot_951.outs  (ctx : parallel_generatedRoot_951Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_951.w_49_0 ctx), ((parallel_generatedRoot_951.w_61_0 ctx), ((parallel_generatedRoot_951.w_63_0 ctx), ((parallel_generatedRoot_951.w_65_0 ctx), ((parallel_generatedRoot_951.w_69_0 ctx), ((parallel_generatedRoot_951.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_951.Facts_0  (ctx : parallel_generatedRoot_951Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_951.w_1_0 ctx) (0) (parallel_generatedRoot_951.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_951.w_3_0 ctx) (0) (parallel_generatedRoot_951.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_951.w_5_0 ctx) (0) (parallel_generatedRoot_951.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_951.w_7_0 ctx) (0) (parallel_generatedRoot_951.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_951_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_951.w_9_0 ctx), ((parallel_generatedRoot_951.w_10_0 ctx), ((parallel_generatedRoot_951.w_11_0 ctx), ((parallel_generatedRoot_951.w_12_0 ctx), ((parallel_generatedRoot_951.w_13_0 ctx), ((parallel_generatedRoot_951.w_14_0 ctx), ((parallel_generatedRoot_951.w_15_0 ctx), ((parallel_generatedRoot_951.w_16_0 ctx), ((parallel_generatedRoot_951.w_17_0 ctx), ((parallel_generatedRoot_951.w_1_0 ctx), ((parallel_generatedRoot_951.w_3_0 ctx), ((parallel_generatedRoot_951.w_5_0 ctx), ((parallel_generatedRoot_951.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_951.w_0_0 ctx), ((parallel_generatedRoot_951.w_0_0 ctx), ((parallel_generatedRoot_951.w_0_0 ctx), ((parallel_generatedRoot_951.w_0_0 ctx), ((parallel_generatedRoot_951.w_0_0 ctx), ((parallel_generatedRoot_951.w_0_0 ctx), ((parallel_generatedRoot_951.w_2_0 ctx), ((parallel_generatedRoot_951.w_4_0 ctx), ((parallel_generatedRoot_951.w_6_0 ctx), ((parallel_generatedRoot_951.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_951.w_18_0 ctx), ((parallel_generatedRoot_951.w_18_1 ctx), ((parallel_generatedRoot_951.w_18_2 ctx), ((parallel_generatedRoot_951.w_18_3 ctx), ((parallel_generatedRoot_951.w_18_4 ctx), ((parallel_generatedRoot_951.w_18_5 ctx), ((parallel_generatedRoot_951.w_18_6 ctx), ((parallel_generatedRoot_951.w_18_7 ctx), ((parallel_generatedRoot_951.w_18_8 ctx), ((parallel_generatedRoot_951.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_951.w_9_0 ctx) (7) (parallel_generatedRoot_951.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_951.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_951.w_10_0 ctx)) (parallel_generatedRoot_951.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_951.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_951.w_10_0 ctx)) (parallel_generatedRoot_951.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_19_0 ctx) (parallel_generatedRoot_951.w_21_0 ctx) (parallel_generatedRoot_951.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_23_concat_1 ctx) (parallel_generatedRoot_951.w_22_0 ctx) (parallel_generatedRoot_951.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_951.w_11_0 ctx) (7) (parallel_generatedRoot_951.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_951.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_951.w_10_0 ctx)) (parallel_generatedRoot_951.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_951.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_951.w_10_0 ctx)) (parallel_generatedRoot_951.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_24_0 ctx) (parallel_generatedRoot_951.w_25_0 ctx) (parallel_generatedRoot_951.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_27_concat_1 ctx) (parallel_generatedRoot_951.w_26_0 ctx) (parallel_generatedRoot_951.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_951.w_23_0 ctx) (parallel_generatedRoot_951.w_27_0 ctx) (parallel_generatedRoot_951.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_951.w_12_0 ctx) (parallel_generatedRoot_951.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_951.w_13_0 ctx) (parallel_generatedRoot_951.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_951.w_14_0 ctx) (parallel_generatedRoot_951.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_29_0 ctx) (parallel_generatedRoot_951.w_30_0 ctx) (parallel_generatedRoot_951.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_32_concat_1 ctx) (parallel_generatedRoot_951.w_31_0 ctx) (parallel_generatedRoot_951.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_951.facts_0  {ctx : parallel_generatedRoot_951Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_951.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_951.w_1_0 ctx) (parallel_generatedRoot_951.w_3_0 ctx) (parallel_generatedRoot_951.w_5_0 ctx) (parallel_generatedRoot_951.w_7_0 ctx) (parallel_generatedRoot_951.w_9_0 ctx) (parallel_generatedRoot_951.w_10_0 ctx) (parallel_generatedRoot_951.w_11_0 ctx) (parallel_generatedRoot_951.w_12_0 ctx) (parallel_generatedRoot_951.w_13_0 ctx) (parallel_generatedRoot_951.w_14_0 ctx) (parallel_generatedRoot_951.w_15_0 ctx) (parallel_generatedRoot_951.w_16_0 ctx) (parallel_generatedRoot_951.w_17_0 ctx) (parallel_generatedRoot_951.w_0_0 ctx) (parallel_generatedRoot_951.w_2_0 ctx) (parallel_generatedRoot_951.w_4_0 ctx) (parallel_generatedRoot_951.w_6_0 ctx) (parallel_generatedRoot_951.w_8_0 ctx) (parallel_generatedRoot_951.w_18_0 ctx) (parallel_generatedRoot_951.w_18_1 ctx) (parallel_generatedRoot_951.w_18_2 ctx) (parallel_generatedRoot_951.w_18_3 ctx) (parallel_generatedRoot_951.w_18_4 ctx) (parallel_generatedRoot_951.w_18_5 ctx) (parallel_generatedRoot_951.w_18_6 ctx) (parallel_generatedRoot_951.w_18_7 ctx) (parallel_generatedRoot_951.w_18_8 ctx) (parallel_generatedRoot_951.w_18_9 ctx) (parallel_generatedRoot_951.w_19_0 ctx) (parallel_generatedRoot_951.w_20_0 ctx) (parallel_generatedRoot_951.w_21_0 ctx) (parallel_generatedRoot_951.w_22_0 ctx) (parallel_generatedRoot_951.w_23_0 ctx) (parallel_generatedRoot_951.w_23_concat_1 ctx) (parallel_generatedRoot_951.w_24_0 ctx) (parallel_generatedRoot_951.w_25_0 ctx) (parallel_generatedRoot_951.w_26_0 ctx) (parallel_generatedRoot_951.w_27_0 ctx) (parallel_generatedRoot_951.w_27_concat_1 ctx) (parallel_generatedRoot_951.w_28_0 ctx) (parallel_generatedRoot_951.w_29_0 ctx) (parallel_generatedRoot_951.w_30_0 ctx) (parallel_generatedRoot_951.w_31_0 ctx) (parallel_generatedRoot_951.w_32_0 ctx) (parallel_generatedRoot_951.w_32_concat_1 ctx)) :
    parallel_generatedRoot_951.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_951.Facts_1  (ctx : parallel_generatedRoot_951Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_951.w_15_0 ctx) (parallel_generatedRoot_951.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_951.w_16_0 ctx) (parallel_generatedRoot_951.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_951.w_17_0 ctx) (parallel_generatedRoot_951.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_33_0 ctx) (parallel_generatedRoot_951.w_34_0 ctx) (parallel_generatedRoot_951.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_36_concat_1 ctx) (parallel_generatedRoot_951.w_35_0 ctx) (parallel_generatedRoot_951.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_951.w_32_0 ctx) (parallel_generatedRoot_951.w_36_0 ctx) (parallel_generatedRoot_951.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_18_6 ctx) (parallel_generatedRoot_951.w_18_8 ctx) (parallel_generatedRoot_951.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_951.w_38_0 ctx) (parallel_generatedRoot_951.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_951.w_39_0 ctx) (parallel_generatedRoot_951.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_951.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_951.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_42_0 ctx) (parallel_generatedRoot_951.w_43_0 ctx) (parallel_generatedRoot_951.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_951.w_44_0 ctx) (parallel_generatedRoot_951.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_951.w_45_0 ctx) (parallel_generatedRoot_951.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_951.facts_1  {ctx : parallel_generatedRoot_951Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_951.constraints_1 backend (parallel_generatedRoot_951.w_15_0 ctx) (parallel_generatedRoot_951.w_16_0 ctx) (parallel_generatedRoot_951.w_17_0 ctx) (parallel_generatedRoot_951.w_18_6 ctx) (parallel_generatedRoot_951.w_18_8 ctx) (parallel_generatedRoot_951.w_32_0 ctx) (parallel_generatedRoot_951.w_33_0 ctx) (parallel_generatedRoot_951.w_34_0 ctx) (parallel_generatedRoot_951.w_35_0 ctx) (parallel_generatedRoot_951.w_36_0 ctx) (parallel_generatedRoot_951.w_36_concat_1 ctx) (parallel_generatedRoot_951.w_37_0 ctx) (parallel_generatedRoot_951.w_38_0 ctx) (parallel_generatedRoot_951.w_39_0 ctx) (parallel_generatedRoot_951.w_40_0 ctx) (parallel_generatedRoot_951.w_41_0 ctx) (parallel_generatedRoot_951.w_42_0 ctx) (parallel_generatedRoot_951.w_43_0 ctx) (parallel_generatedRoot_951.w_44_0 ctx) (parallel_generatedRoot_951.w_45_0 ctx) (parallel_generatedRoot_951.w_46_0 ctx)) :
    parallel_generatedRoot_951.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_951.Facts_2  (ctx : parallel_generatedRoot_951Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_951.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_18_7 ctx) (parallel_generatedRoot_951.w_18_9 ctx) (parallel_generatedRoot_951.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_951.w_50_0 ctx) (parallel_generatedRoot_951.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_951.w_51_0 ctx) (parallel_generatedRoot_951.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_951.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_951.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_951.w_54_0 ctx) (parallel_generatedRoot_951.w_55_0 ctx) (parallel_generatedRoot_951.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_951.w_56_0 ctx) (parallel_generatedRoot_951.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_951.w_57_0 ctx) (parallel_generatedRoot_951.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_951.w_60_0 ctx)

theorem parallel_generatedRoot_951.facts_2  {ctx : parallel_generatedRoot_951Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_951.constraints_2 backend (parallel_generatedRoot_951.w_18_7 ctx) (parallel_generatedRoot_951.w_18_9 ctx) (parallel_generatedRoot_951.w_47_0 ctx) (parallel_generatedRoot_951.w_48_0 ctx) (parallel_generatedRoot_951.w_50_0 ctx) (parallel_generatedRoot_951.w_51_0 ctx) (parallel_generatedRoot_951.w_52_0 ctx) (parallel_generatedRoot_951.w_53_0 ctx) (parallel_generatedRoot_951.w_54_0 ctx) (parallel_generatedRoot_951.w_55_0 ctx) (parallel_generatedRoot_951.w_56_0 ctx) (parallel_generatedRoot_951.w_57_0 ctx) (parallel_generatedRoot_951.w_58_0 ctx) (parallel_generatedRoot_951.w_59_0 ctx) (parallel_generatedRoot_951.w_60_0 ctx)) :
    parallel_generatedRoot_951.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_951.Facts_3  (ctx : parallel_generatedRoot_951Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_951.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_951.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_951.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_951.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_951.facts_3  {ctx : parallel_generatedRoot_951Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_951.constraints_3 (parallel_generatedRoot_951.w_41_0 ctx) (parallel_generatedRoot_951.w_47_0 ctx) (parallel_generatedRoot_951.w_59_0 ctx) (parallel_generatedRoot_951.w_62_0 ctx) (parallel_generatedRoot_951.w_64_0 ctx) (parallel_generatedRoot_951.w_66_0 ctx) (parallel_generatedRoot_951.w_67_0 ctx)) :
    parallel_generatedRoot_951.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_951.Facts_4  (ctx : parallel_generatedRoot_951Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_951.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_951.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_951.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_951.outs ctx

theorem parallel_generatedRoot_951.facts_4  {ctx : parallel_generatedRoot_951Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_951.constraints_4 (parallel_generatedRoot_951.w_49_0 ctx) (parallel_generatedRoot_951.w_53_0 ctx) (parallel_generatedRoot_951.w_59_0 ctx) (parallel_generatedRoot_951.w_61_0 ctx) (parallel_generatedRoot_951.w_63_0 ctx) (parallel_generatedRoot_951.w_65_0 ctx) (parallel_generatedRoot_951.w_69_0 ctx) (parallel_generatedRoot_951.w_70_0 ctx) (parallel_generatedRoot_951.w_71_0 ctx) (parallel_generatedRoot_951.w_73_0 ctx) outputs) :
    parallel_generatedRoot_951.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_951.Facts  (ctx : parallel_generatedRoot_951Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_951.Facts_0 ctx
  c1 : parallel_generatedRoot_951.Facts_1 ctx
  c2 : parallel_generatedRoot_951.Facts_2 ctx
  c3 : parallel_generatedRoot_951.Facts_3 ctx
  c4 : parallel_generatedRoot_951.Facts_4 ctx outputs

theorem parallel_generatedRoot_951.facts  (ctx : parallel_generatedRoot_951Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_951.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_951.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_951.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_951.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_951.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_951.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_951.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_983.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (sequential_generatedRoot_983.InputType)
  witness : sequential_generatedRoot_983.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_983Ctx" => (sequential_generatedRoot_983.Ctx backend hashModel params i_0)

def sequential_generatedRoot_983.w_0_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.inputs.1

def sequential_generatedRoot_983.w_31_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.inputs.2.1

def sequential_generatedRoot_983.w_35_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_983.w_39_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_983.w_43_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_983.w_47_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_983.w_2_0  (ctx : sequential_generatedRoot_983Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_983.w_8_0  (ctx : sequential_generatedRoot_983Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_983.w_12_0  (ctx : sequential_generatedRoot_983Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_983.w_16_0  (ctx : sequential_generatedRoot_983Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_983.w_20_0  (ctx : sequential_generatedRoot_983Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_983.w_24_0  (ctx : sequential_generatedRoot_983Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_983.w_1_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_983.w_3_0  (ctx : sequential_generatedRoot_983Ctx) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_983.w_4_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_983.w_5_0  (ctx : sequential_generatedRoot_983Ctx) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_983.w_6_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_983.w_7_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_4_0 ctx) (sequential_generatedRoot_983.w_6_0 ctx)

def sequential_generatedRoot_983.w_9_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_983.w_10_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_983.w_11_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_9_0 ctx) (sequential_generatedRoot_983.w_10_0 ctx)

def sequential_generatedRoot_983.w_13_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_983.w_14_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_983.w_15_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_13_0 ctx) (sequential_generatedRoot_983.w_14_0 ctx)

def sequential_generatedRoot_983.w_17_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_983.w_18_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_983.w_19_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_17_0 ctx) (sequential_generatedRoot_983.w_18_0 ctx)

def sequential_generatedRoot_983.w_21_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_983.w_22_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def sequential_generatedRoot_983.w_23_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_21_0 ctx) (sequential_generatedRoot_983.w_22_0 ctx)

def sequential_generatedRoot_983.w_25_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_983.w_26_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def sequential_generatedRoot_983.w_27_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_25_0 ctx) (sequential_generatedRoot_983.w_26_0 ctx)

def sequential_generatedRoot_983.w_28_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_28_0

def sequential_generatedRoot_983.w_28_1  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_28_1

def sequential_generatedRoot_983.w_28_2  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_28_2

def sequential_generatedRoot_983.w_28_3  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_28_3

def sequential_generatedRoot_983.w_28_4  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_28_4

def sequential_generatedRoot_983.w_28_5  (ctx : sequential_generatedRoot_983Ctx) : mxxMat0 :=
  ctx.witness.w_28_5

def sequential_generatedRoot_983.w_29_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_983.w_30_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_0_0 ctx) (sequential_generatedRoot_983.w_29_0 ctx)

def sequential_generatedRoot_983.w_32_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_983.w_33_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_983.w_34_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_31_0 ctx) (sequential_generatedRoot_983.w_33_0 ctx)

def sequential_generatedRoot_983.w_36_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_983.w_37_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.witness.w_37_0

def sequential_generatedRoot_983.w_38_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_35_0 ctx) (sequential_generatedRoot_983.w_37_0 ctx)

def sequential_generatedRoot_983.w_40_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_983.w_41_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.witness.w_41_0

def sequential_generatedRoot_983.w_42_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_39_0 ctx) (sequential_generatedRoot_983.w_41_0 ctx)

def sequential_generatedRoot_983.w_44_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_983.w_45_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.witness.w_45_0

def sequential_generatedRoot_983.w_46_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_43_0 ctx) (sequential_generatedRoot_983.w_45_0 ctx)

def sequential_generatedRoot_983.w_48_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_983.w_49_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  ctx.witness.w_49_0

def sequential_generatedRoot_983.w_50_0  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_983.w_47_0 ctx) (sequential_generatedRoot_983.w_49_0 ctx)

def sequential_generatedRoot_983.outs  (ctx : sequential_generatedRoot_983Ctx) : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit :=
  ((sequential_generatedRoot_983.w_30_0 ctx), ((sequential_generatedRoot_983.w_34_0 ctx), ((sequential_generatedRoot_983.w_38_0 ctx), ((sequential_generatedRoot_983.w_42_0 ctx), ((sequential_generatedRoot_983.w_46_0 ctx), ((sequential_generatedRoot_983.w_50_0 ctx), ()))))))

structure sequential_generatedRoot_983.Facts_0  (ctx : sequential_generatedRoot_983Ctx) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 4500
  h3_0 : 0 ≤ (sequential_generatedRoot_983.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_983.w_3_0 ctx) < 36000
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_2_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_983.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_983.w_5_0 ctx) < 36000
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_2_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_983.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_983.w_3_0 ctx) < 36000
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_8_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_983.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_983.w_5_0 ctx) < 36000
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_8_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_983.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_983.w_3_0 ctx) < 36000
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_12_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_983.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_983.w_5_0 ctx) < 36000
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_12_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_983.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_983.w_3_0 ctx) < 36000
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_16_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_983.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_983.w_5_0 ctx) < 36000
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_16_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_18_0 ctx)
  h3_0_8 : 0 ≤ (sequential_generatedRoot_983.w_3_0 ctx)
  h3_0_9 : (sequential_generatedRoot_983.w_3_0 ctx) < 36000
  h21_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_20_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_21_0 ctx)
  h5_0_8 : 0 ≤ (sequential_generatedRoot_983.w_5_0 ctx)
  h5_0_9 : (sequential_generatedRoot_983.w_5_0 ctx) < 36000
  h22_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_20_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_22_0 ctx)

theorem sequential_generatedRoot_983.facts_0  {ctx : sequential_generatedRoot_983Ctx} {outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit}
    (h : sequential_generatedRoot_983.constraints_0 i_0 (sequential_generatedRoot_983.w_2_0 ctx) (sequential_generatedRoot_983.w_8_0 ctx) (sequential_generatedRoot_983.w_12_0 ctx) (sequential_generatedRoot_983.w_16_0 ctx) (sequential_generatedRoot_983.w_20_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_4_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_6_0 ctx) (sequential_generatedRoot_983.w_9_0 ctx) (sequential_generatedRoot_983.w_10_0 ctx) (sequential_generatedRoot_983.w_13_0 ctx) (sequential_generatedRoot_983.w_14_0 ctx) (sequential_generatedRoot_983.w_17_0 ctx) (sequential_generatedRoot_983.w_18_0 ctx) (sequential_generatedRoot_983.w_21_0 ctx) (sequential_generatedRoot_983.w_22_0 ctx)) :
    sequential_generatedRoot_983.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_983.Facts_1  (ctx : sequential_generatedRoot_983Ctx) (outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) : Prop where
  h3_0_10 : 0 ≤ (sequential_generatedRoot_983.w_3_0 ctx)
  h3_0_11 : (sequential_generatedRoot_983.w_3_0 ctx) < 36000
  h25_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_24_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_25_0 ctx)
  h5_0_10 : 0 ≤ (sequential_generatedRoot_983.w_5_0 ctx)
  h5_0_11 : (sequential_generatedRoot_983.w_5_0 ctx) < 36000
  h26_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_983.w_24_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_26_0 ctx)
  t38 : 0 ≤ (6)
  h28_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_89 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((sequential_generatedRoot_983.w_2_0 ctx), ((sequential_generatedRoot_983.w_8_0 ctx), ((sequential_generatedRoot_983.w_12_0 ctx), ((sequential_generatedRoot_983.w_16_0 ctx), ((sequential_generatedRoot_983.w_20_0 ctx), ((sequential_generatedRoot_983.w_24_0 ctx), ())))))))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_983.w_7_0 ctx), ((sequential_generatedRoot_983.w_11_0 ctx), ((sequential_generatedRoot_983.w_15_0 ctx), ((sequential_generatedRoot_983.w_19_0 ctx), ((sequential_generatedRoot_983.w_23_0 ctx), ((sequential_generatedRoot_983.w_27_0 ctx), ())))))) ((sequential_generatedRoot_983.w_28_0 ctx), ((sequential_generatedRoot_983.w_28_1 ctx), ((sequential_generatedRoot_983.w_28_2 ctx), ((sequential_generatedRoot_983.w_28_3 ctx), ((sequential_generatedRoot_983.w_28_4 ctx), ((sequential_generatedRoot_983.w_28_5 ctx), ()))))))
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_983.w_1_0 ctx) (sequential_generatedRoot_983.w_28_0 ctx) (sequential_generatedRoot_983.w_29_0 ctx)
  t41 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t42 : ((1 + (Int.ofNat i_0))) < 4500
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_983.w_32_0 ctx) (sequential_generatedRoot_983.w_28_1 ctx) (sequential_generatedRoot_983.w_33_0 ctx)
  t44 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t45 : ((1 + (Int.ofNat i_0))) < 4500
  h37_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_983.w_36_0 ctx) (sequential_generatedRoot_983.w_28_2 ctx) (sequential_generatedRoot_983.w_37_0 ctx)
  t47 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t48 : ((1 + (Int.ofNat i_0))) < 4500
  h41_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_983.w_40_0 ctx) (sequential_generatedRoot_983.w_28_3 ctx) (sequential_generatedRoot_983.w_41_0 ctx)
  t50 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t51 : ((1 + (Int.ofNat i_0))) < 4500
  h45_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_983.w_44_0 ctx) (sequential_generatedRoot_983.w_28_4 ctx) (sequential_generatedRoot_983.w_45_0 ctx)
  t53 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t54 : ((1 + (Int.ofNat i_0))) < 4500
  h49_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_983.w_48_0 ctx) (sequential_generatedRoot_983.w_28_5 ctx) (sequential_generatedRoot_983.w_49_0 ctx)
  hout : outputs = sequential_generatedRoot_983.outs ctx

theorem sequential_generatedRoot_983.facts_1  {ctx : sequential_generatedRoot_983Ctx} {outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit}
    (h : sequential_generatedRoot_983.constraints_1 backend hashModel params i_0 (sequential_generatedRoot_983.w_2_0 ctx) (sequential_generatedRoot_983.w_8_0 ctx) (sequential_generatedRoot_983.w_12_0 ctx) (sequential_generatedRoot_983.w_16_0 ctx) (sequential_generatedRoot_983.w_20_0 ctx) (sequential_generatedRoot_983.w_24_0 ctx) (sequential_generatedRoot_983.w_1_0 ctx) (sequential_generatedRoot_983.w_3_0 ctx) (sequential_generatedRoot_983.w_5_0 ctx) (sequential_generatedRoot_983.w_7_0 ctx) (sequential_generatedRoot_983.w_11_0 ctx) (sequential_generatedRoot_983.w_15_0 ctx) (sequential_generatedRoot_983.w_19_0 ctx) (sequential_generatedRoot_983.w_23_0 ctx) (sequential_generatedRoot_983.w_25_0 ctx) (sequential_generatedRoot_983.w_26_0 ctx) (sequential_generatedRoot_983.w_27_0 ctx) (sequential_generatedRoot_983.w_28_0 ctx) (sequential_generatedRoot_983.w_28_1 ctx) (sequential_generatedRoot_983.w_28_2 ctx) (sequential_generatedRoot_983.w_28_3 ctx) (sequential_generatedRoot_983.w_28_4 ctx) (sequential_generatedRoot_983.w_28_5 ctx) (sequential_generatedRoot_983.w_29_0 ctx) (sequential_generatedRoot_983.w_30_0 ctx) (sequential_generatedRoot_983.w_32_0 ctx) (sequential_generatedRoot_983.w_33_0 ctx) (sequential_generatedRoot_983.w_34_0 ctx) (sequential_generatedRoot_983.w_36_0 ctx) (sequential_generatedRoot_983.w_37_0 ctx) (sequential_generatedRoot_983.w_38_0 ctx) (sequential_generatedRoot_983.w_40_0 ctx) (sequential_generatedRoot_983.w_41_0 ctx) (sequential_generatedRoot_983.w_42_0 ctx) (sequential_generatedRoot_983.w_44_0 ctx) (sequential_generatedRoot_983.w_45_0 ctx) (sequential_generatedRoot_983.w_46_0 ctx) (sequential_generatedRoot_983.w_48_0 ctx) (sequential_generatedRoot_983.w_49_0 ctx) (sequential_generatedRoot_983.w_50_0 ctx) outputs) :
    sequential_generatedRoot_983.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_983.Facts  (ctx : sequential_generatedRoot_983Ctx) (outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) : Prop where
  c0 : sequential_generatedRoot_983.Facts_0 ctx
  c1 : sequential_generatedRoot_983.Facts_1 ctx outputs

theorem sequential_generatedRoot_983.facts  (ctx : sequential_generatedRoot_983Ctx) {outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit}
    (h : sequential_generatedRoot_983.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_983.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_983.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_983.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_991.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_991.InputType)
  witness : parallel_generatedRoot_991.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_991Ctx" => (parallel_generatedRoot_991.Ctx backend hashModel params i_0)

def parallel_generatedRoot_991.w_0_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat11 :=
  ctx.inputs.1

def parallel_generatedRoot_991.w_1_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat12 :=
  ctx.inputs.2.1

def parallel_generatedRoot_991.w_4_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_991.w_5_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_991.w_8_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat2 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_991.w_13_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_991.w_29_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_31_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_48_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_49_0  (ctx : parallel_generatedRoot_991Ctx) : Fin 1 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_50_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_51_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_55_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_61_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat12 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_111_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat12 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_161_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat12 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_211_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat12 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_991.w_2_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_0_0 ctx) (parallel_generatedRoot_991.w_1_0 ctx)

def parallel_generatedRoot_991.w_3_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_3_0

def parallel_generatedRoot_991.w_6_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat13 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_991.w_7_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)

def parallel_generatedRoot_991.w_9_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_991.w_7_0 ctx) (parallel_generatedRoot_991.w_8_0 ctx)

def parallel_generatedRoot_991.w_10_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_991.w_11_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_991.w_12_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_6_0 ctx) (parallel_generatedRoot_991.w_11_0 ctx)

def parallel_generatedRoot_991.w_14_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_991.w_15_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_991.w_16_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat14 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_991.w_17_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat15 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat15)

def parallel_generatedRoot_991.w_18_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat14 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_16_0 ctx) (parallel_generatedRoot_991.w_17_0 ctx)

def parallel_generatedRoot_991.w_19_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat15 :=
  (MxxRuntime.matrixPolynomial [(11560550400)] : mxxMat15)

def parallel_generatedRoot_991.w_20_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat14 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_18_0 ctx) (parallel_generatedRoot_991.w_19_0 ctx)

def parallel_generatedRoot_991.w_21_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_991.w_22_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat5)

def parallel_generatedRoot_991.w_23_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_21_0 ctx) (parallel_generatedRoot_991.w_22_0 ctx)

def parallel_generatedRoot_991.w_24_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_991.w_25_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_991.w_26_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_13_0 ctx) (parallel_generatedRoot_991.w_25_0 ctx)

def parallel_generatedRoot_991.w_27_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_12_0 ctx) (parallel_generatedRoot_991.w_26_0 ctx)

def parallel_generatedRoot_991.w_28_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_991.w_30_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_991.w_32_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_991.w_33_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_991.w_34_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_991.w_35_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_33_0 ctx) (parallel_generatedRoot_991.w_34_0 ctx)

def parallel_generatedRoot_991.w_36_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_991.w_37_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_991.w_38_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_30_0 ctx) (parallel_generatedRoot_991.w_37_0 ctx)

def parallel_generatedRoot_991.w_39_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_991.w_40_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)

def parallel_generatedRoot_991.w_41_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_41_0

def parallel_generatedRoot_991.w_42_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_991.w_43_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_39_0 ctx) (parallel_generatedRoot_991.w_42_0 ctx)

def parallel_generatedRoot_991.w_44_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_991.w_45_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_991.w_46_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_27_0 ctx) (parallel_generatedRoot_991.w_45_0 ctx)

def parallel_generatedRoot_991.w_47_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_991.w_52_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_991.w_52_1  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_52_1

def parallel_generatedRoot_991.w_52_2  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_52_2

def parallel_generatedRoot_991.w_53_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_46_0 ctx) (parallel_generatedRoot_991.w_52_0 ctx)

def parallel_generatedRoot_991.w_54_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_3_0 ctx) (parallel_generatedRoot_991.w_53_0 ctx)

def parallel_generatedRoot_991.w_56_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_54_0 ctx) (parallel_generatedRoot_991.w_55_0 ctx)

def parallel_generatedRoot_991.w_57_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat14 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_991.w_58_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_991.w_59_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat5)

def parallel_generatedRoot_991.w_60_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_58_0 ctx) (parallel_generatedRoot_991.w_59_0 ctx)

def parallel_generatedRoot_991.w_62_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_0_0 ctx) (parallel_generatedRoot_991.w_61_0 ctx)

def parallel_generatedRoot_991.w_63_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_63_0

def parallel_generatedRoot_991.w_64_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat13 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_991.w_65_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)

def parallel_generatedRoot_991.w_66_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_991.w_65_0 ctx) (parallel_generatedRoot_991.w_8_0 ctx)

def parallel_generatedRoot_991.w_67_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_991.w_68_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_68_0

def parallel_generatedRoot_991.w_69_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_64_0 ctx) (parallel_generatedRoot_991.w_68_0 ctx)

def parallel_generatedRoot_991.w_70_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_991.w_71_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_991.w_72_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat20 :=
  ctx.witness.w_72_0

def parallel_generatedRoot_991.w_73_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat21 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat21)

def parallel_generatedRoot_991.w_74_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat20 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_72_0 ctx) (parallel_generatedRoot_991.w_73_0 ctx)

def parallel_generatedRoot_991.w_75_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat21 :=
  (MxxRuntime.matrixPolynomial [(1125894775373377)] : mxxMat21)

def parallel_generatedRoot_991.w_76_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat20 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_74_0 ctx) (parallel_generatedRoot_991.w_75_0 ctx)

def parallel_generatedRoot_991.w_77_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_77_0

def parallel_generatedRoot_991.w_78_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat5)

def parallel_generatedRoot_991.w_79_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_77_0 ctx) (parallel_generatedRoot_991.w_78_0 ctx)

def parallel_generatedRoot_991.w_80_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_80_0

def parallel_generatedRoot_991.w_81_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_81_0

def parallel_generatedRoot_991.w_82_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_13_0 ctx) (parallel_generatedRoot_991.w_81_0 ctx)

def parallel_generatedRoot_991.w_83_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_69_0 ctx) (parallel_generatedRoot_991.w_82_0 ctx)

def parallel_generatedRoot_991.w_84_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_991.w_85_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  ctx.witness.w_85_0

def parallel_generatedRoot_991.w_86_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_86_0

def parallel_generatedRoot_991.w_87_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_87_0

def parallel_generatedRoot_991.w_88_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_991.w_89_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_87_0 ctx) (parallel_generatedRoot_991.w_88_0 ctx)

def parallel_generatedRoot_991.w_90_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_90_0

def parallel_generatedRoot_991.w_91_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_91_0

def parallel_generatedRoot_991.w_92_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_85_0 ctx) (parallel_generatedRoot_991.w_91_0 ctx)

def parallel_generatedRoot_991.w_93_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_93_0

def parallel_generatedRoot_991.w_94_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)

def parallel_generatedRoot_991.w_95_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_95_0

def parallel_generatedRoot_991.w_96_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_96_0

def parallel_generatedRoot_991.w_97_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_93_0 ctx) (parallel_generatedRoot_991.w_96_0 ctx)

def parallel_generatedRoot_991.w_98_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_98_0

def parallel_generatedRoot_991.w_99_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_99_0

def parallel_generatedRoot_991.w_100_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_83_0 ctx) (parallel_generatedRoot_991.w_99_0 ctx)

def parallel_generatedRoot_991.w_101_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_991.w_102_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_102_0

def parallel_generatedRoot_991.w_102_1  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_102_1

def parallel_generatedRoot_991.w_102_2  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_102_2

def parallel_generatedRoot_991.w_103_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_100_0 ctx) (parallel_generatedRoot_991.w_102_0 ctx)

def parallel_generatedRoot_991.w_104_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_63_0 ctx) (parallel_generatedRoot_991.w_103_0 ctx)

def parallel_generatedRoot_991.w_105_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_104_0 ctx) (parallel_generatedRoot_991.w_55_0 ctx)

def parallel_generatedRoot_991.w_106_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat20 :=
  ctx.witness.w_106_0

def parallel_generatedRoot_991.w_107_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_107_0

def parallel_generatedRoot_991.w_108_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat5)

def parallel_generatedRoot_991.w_109_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_107_0 ctx) (parallel_generatedRoot_991.w_108_0 ctx)

def parallel_generatedRoot_991.w_110_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_60_0 ctx) (parallel_generatedRoot_991.w_109_0 ctx)

def parallel_generatedRoot_991.w_112_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_0_0 ctx) (parallel_generatedRoot_991.w_111_0 ctx)

def parallel_generatedRoot_991.w_113_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_113_0

def parallel_generatedRoot_991.w_114_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat13 :=
  ctx.witness.w_114_0

def parallel_generatedRoot_991.w_115_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)

def parallel_generatedRoot_991.w_116_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_991.w_115_0 ctx) (parallel_generatedRoot_991.w_8_0 ctx)

def parallel_generatedRoot_991.w_117_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_117_0

def parallel_generatedRoot_991.w_118_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_118_0

def parallel_generatedRoot_991.w_119_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_114_0 ctx) (parallel_generatedRoot_991.w_118_0 ctx)

def parallel_generatedRoot_991.w_120_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_120_0

def parallel_generatedRoot_991.w_121_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_121_0

def parallel_generatedRoot_991.w_122_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat22 :=
  ctx.witness.w_122_0

def parallel_generatedRoot_991.w_123_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat23 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat23)

def parallel_generatedRoot_991.w_124_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat22 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_122_0 ctx) (parallel_generatedRoot_991.w_123_0 ctx)

def parallel_generatedRoot_991.w_125_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat23 :=
  (MxxRuntime.matrixPolynomial [(7254835200)] : mxxMat23)

def parallel_generatedRoot_991.w_126_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat22 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_124_0 ctx) (parallel_generatedRoot_991.w_125_0 ctx)

def parallel_generatedRoot_991.w_127_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_127_0

def parallel_generatedRoot_991.w_128_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat5)

def parallel_generatedRoot_991.w_129_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_127_0 ctx) (parallel_generatedRoot_991.w_128_0 ctx)

def parallel_generatedRoot_991.w_130_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_130_0

def parallel_generatedRoot_991.w_131_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_131_0

def parallel_generatedRoot_991.w_132_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_13_0 ctx) (parallel_generatedRoot_991.w_131_0 ctx)

def parallel_generatedRoot_991.w_133_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_119_0 ctx) (parallel_generatedRoot_991.w_132_0 ctx)

def parallel_generatedRoot_991.w_134_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_991.w_135_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  ctx.witness.w_135_0

def parallel_generatedRoot_991.w_136_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_136_0

def parallel_generatedRoot_991.w_137_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_137_0

def parallel_generatedRoot_991.w_138_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_991.w_139_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_137_0 ctx) (parallel_generatedRoot_991.w_138_0 ctx)

def parallel_generatedRoot_991.w_140_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_140_0

def parallel_generatedRoot_991.w_141_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_141_0

def parallel_generatedRoot_991.w_142_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_135_0 ctx) (parallel_generatedRoot_991.w_141_0 ctx)

def parallel_generatedRoot_991.w_143_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_143_0

def parallel_generatedRoot_991.w_144_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)

def parallel_generatedRoot_991.w_145_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_145_0

def parallel_generatedRoot_991.w_146_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_146_0

def parallel_generatedRoot_991.w_147_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_143_0 ctx) (parallel_generatedRoot_991.w_146_0 ctx)

def parallel_generatedRoot_991.w_148_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_148_0

def parallel_generatedRoot_991.w_149_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_149_0

def parallel_generatedRoot_991.w_150_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_133_0 ctx) (parallel_generatedRoot_991.w_149_0 ctx)

def parallel_generatedRoot_991.w_151_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_991.w_152_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_152_0

def parallel_generatedRoot_991.w_152_1  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_152_1

def parallel_generatedRoot_991.w_152_2  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_152_2

def parallel_generatedRoot_991.w_153_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_150_0 ctx) (parallel_generatedRoot_991.w_152_0 ctx)

def parallel_generatedRoot_991.w_154_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_113_0 ctx) (parallel_generatedRoot_991.w_153_0 ctx)

def parallel_generatedRoot_991.w_155_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_154_0 ctx) (parallel_generatedRoot_991.w_55_0 ctx)

def parallel_generatedRoot_991.w_156_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat22 :=
  ctx.witness.w_156_0

def parallel_generatedRoot_991.w_157_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_157_0

def parallel_generatedRoot_991.w_158_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat5)

def parallel_generatedRoot_991.w_159_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_157_0 ctx) (parallel_generatedRoot_991.w_158_0 ctx)

def parallel_generatedRoot_991.w_160_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_110_0 ctx) (parallel_generatedRoot_991.w_159_0 ctx)

def parallel_generatedRoot_991.w_162_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_0_0 ctx) (parallel_generatedRoot_991.w_161_0 ctx)

def parallel_generatedRoot_991.w_163_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_163_0

def parallel_generatedRoot_991.w_164_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat13 :=
  ctx.witness.w_164_0

def parallel_generatedRoot_991.w_165_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)

def parallel_generatedRoot_991.w_166_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_991.w_165_0 ctx) (parallel_generatedRoot_991.w_8_0 ctx)

def parallel_generatedRoot_991.w_167_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_167_0

def parallel_generatedRoot_991.w_168_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_168_0

def parallel_generatedRoot_991.w_169_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_164_0 ctx) (parallel_generatedRoot_991.w_168_0 ctx)

def parallel_generatedRoot_991.w_170_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_170_0

def parallel_generatedRoot_991.w_171_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_171_0

def parallel_generatedRoot_991.w_172_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat24 :=
  ctx.witness.w_172_0

def parallel_generatedRoot_991.w_173_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat25 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat25)

def parallel_generatedRoot_991.w_174_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat24 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_172_0 ctx) (parallel_generatedRoot_991.w_173_0 ctx)

def parallel_generatedRoot_991.w_175_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat25 :=
  (MxxRuntime.matrixPolynomial [(1125871968844913)] : mxxMat25)

def parallel_generatedRoot_991.w_176_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat24 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_174_0 ctx) (parallel_generatedRoot_991.w_175_0 ctx)

def parallel_generatedRoot_991.w_177_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_177_0

def parallel_generatedRoot_991.w_178_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat5)

def parallel_generatedRoot_991.w_179_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_177_0 ctx) (parallel_generatedRoot_991.w_178_0 ctx)

def parallel_generatedRoot_991.w_180_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_180_0

def parallel_generatedRoot_991.w_181_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_181_0

def parallel_generatedRoot_991.w_182_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_13_0 ctx) (parallel_generatedRoot_991.w_181_0 ctx)

def parallel_generatedRoot_991.w_183_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_169_0 ctx) (parallel_generatedRoot_991.w_182_0 ctx)

def parallel_generatedRoot_991.w_184_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_991.w_185_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  ctx.witness.w_185_0

def parallel_generatedRoot_991.w_186_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_186_0

def parallel_generatedRoot_991.w_187_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_187_0

def parallel_generatedRoot_991.w_188_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_991.w_189_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_187_0 ctx) (parallel_generatedRoot_991.w_188_0 ctx)

def parallel_generatedRoot_991.w_190_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_190_0

def parallel_generatedRoot_991.w_191_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_191_0

def parallel_generatedRoot_991.w_192_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_185_0 ctx) (parallel_generatedRoot_991.w_191_0 ctx)

def parallel_generatedRoot_991.w_193_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_193_0

def parallel_generatedRoot_991.w_194_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)

def parallel_generatedRoot_991.w_195_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_195_0

def parallel_generatedRoot_991.w_196_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_196_0

def parallel_generatedRoot_991.w_197_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_193_0 ctx) (parallel_generatedRoot_991.w_196_0 ctx)

def parallel_generatedRoot_991.w_198_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_198_0

def parallel_generatedRoot_991.w_199_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_199_0

def parallel_generatedRoot_991.w_200_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_183_0 ctx) (parallel_generatedRoot_991.w_199_0 ctx)

def parallel_generatedRoot_991.w_201_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_991.w_202_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_202_0

def parallel_generatedRoot_991.w_202_1  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_202_1

def parallel_generatedRoot_991.w_202_2  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_202_2

def parallel_generatedRoot_991.w_203_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_200_0 ctx) (parallel_generatedRoot_991.w_202_0 ctx)

def parallel_generatedRoot_991.w_204_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_163_0 ctx) (parallel_generatedRoot_991.w_203_0 ctx)

def parallel_generatedRoot_991.w_205_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_204_0 ctx) (parallel_generatedRoot_991.w_55_0 ctx)

def parallel_generatedRoot_991.w_206_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat24 :=
  ctx.witness.w_206_0

def parallel_generatedRoot_991.w_207_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_207_0

def parallel_generatedRoot_991.w_208_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat5)

def parallel_generatedRoot_991.w_209_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_207_0 ctx) (parallel_generatedRoot_991.w_208_0 ctx)

def parallel_generatedRoot_991.w_210_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_160_0 ctx) (parallel_generatedRoot_991.w_209_0 ctx)

def parallel_generatedRoot_991.w_212_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_0_0 ctx) (parallel_generatedRoot_991.w_211_0 ctx)

def parallel_generatedRoot_991.w_213_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_213_0

def parallel_generatedRoot_991.w_214_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat13 :=
  ctx.witness.w_214_0

def parallel_generatedRoot_991.w_215_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)

def parallel_generatedRoot_991.w_216_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_991.w_215_0 ctx) (parallel_generatedRoot_991.w_8_0 ctx)

def parallel_generatedRoot_991.w_217_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_217_0

def parallel_generatedRoot_991.w_218_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat4 :=
  ctx.witness.w_218_0

def parallel_generatedRoot_991.w_219_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_214_0 ctx) (parallel_generatedRoot_991.w_218_0 ctx)

def parallel_generatedRoot_991.w_220_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_220_0

def parallel_generatedRoot_991.w_221_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_221_0

def parallel_generatedRoot_991.w_222_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat26 :=
  ctx.witness.w_222_0

def parallel_generatedRoot_991.w_223_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat27 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat27)

def parallel_generatedRoot_991.w_224_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat26 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_222_0 ctx) (parallel_generatedRoot_991.w_223_0 ctx)

def parallel_generatedRoot_991.w_225_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat27 :=
  (MxxRuntime.matrixPolynomial [(157994188800)] : mxxMat27)

def parallel_generatedRoot_991.w_226_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat26 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_224_0 ctx) (parallel_generatedRoot_991.w_225_0 ctx)

def parallel_generatedRoot_991.w_227_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_227_0

def parallel_generatedRoot_991.w_228_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat5)

def parallel_generatedRoot_991.w_229_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_227_0 ctx) (parallel_generatedRoot_991.w_228_0 ctx)

def parallel_generatedRoot_991.w_230_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_230_0

def parallel_generatedRoot_991.w_231_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat6 :=
  ctx.witness.w_231_0

def parallel_generatedRoot_991.w_232_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_13_0 ctx) (parallel_generatedRoot_991.w_231_0 ctx)

def parallel_generatedRoot_991.w_233_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_219_0 ctx) (parallel_generatedRoot_991.w_232_0 ctx)

def parallel_generatedRoot_991.w_234_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_991.w_235_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  ctx.witness.w_235_0

def parallel_generatedRoot_991.w_236_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_236_0

def parallel_generatedRoot_991.w_237_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  ctx.witness.w_237_0

def parallel_generatedRoot_991.w_238_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_991.w_239_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat17 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_237_0 ctx) (parallel_generatedRoot_991.w_238_0 ctx)

def parallel_generatedRoot_991.w_240_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_240_0

def parallel_generatedRoot_991.w_241_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat18 :=
  ctx.witness.w_241_0

def parallel_generatedRoot_991.w_242_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat16 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_235_0 ctx) (parallel_generatedRoot_991.w_241_0 ctx)

def parallel_generatedRoot_991.w_243_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_243_0

def parallel_generatedRoot_991.w_244_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)

def parallel_generatedRoot_991.w_245_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_245_0

def parallel_generatedRoot_991.w_246_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat19 :=
  ctx.witness.w_246_0

def parallel_generatedRoot_991.w_247_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_991.w_243_0 ctx) (parallel_generatedRoot_991.w_246_0 ctx)

def parallel_generatedRoot_991.w_248_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_248_0

def parallel_generatedRoot_991.w_249_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_249_0

def parallel_generatedRoot_991.w_250_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_233_0 ctx) (parallel_generatedRoot_991.w_249_0 ctx)

def parallel_generatedRoot_991.w_251_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_991.w_252_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_252_0

def parallel_generatedRoot_991.w_252_1  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_252_1

def parallel_generatedRoot_991.w_252_2  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_252_2

def parallel_generatedRoot_991.w_253_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_250_0 ctx) (parallel_generatedRoot_991.w_252_0 ctx)

def parallel_generatedRoot_991.w_254_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_213_0 ctx) (parallel_generatedRoot_991.w_253_0 ctx)

def parallel_generatedRoot_991.w_255_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_991.w_254_0 ctx) (parallel_generatedRoot_991.w_55_0 ctx)

def parallel_generatedRoot_991.w_256_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat26 :=
  ctx.witness.w_256_0

def parallel_generatedRoot_991.w_257_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  ctx.witness.w_257_0

def parallel_generatedRoot_991.w_258_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat5)

def parallel_generatedRoot_991.w_259_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_991.w_257_0 ctx) (parallel_generatedRoot_991.w_258_0 ctx)

def parallel_generatedRoot_991.w_260_0  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_991.w_210_0 ctx) (parallel_generatedRoot_991.w_259_0 ctx)

def parallel_generatedRoot_991.outs  (ctx : parallel_generatedRoot_991Ctx) : mxxMat0 :=
  (parallel_generatedRoot_991.w_260_0 ctx)

structure parallel_generatedRoot_991.Facts_0  (ctx : parallel_generatedRoot_991Ctx) : Prop where
  h3_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_2_0 ctx) (parallel_generatedRoot_991.w_3_0 ctx)
  h6_0 : MxxRuntime.concatColumns (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_6_0 ctx)
  h10_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_9_0 ctx) (parallel_generatedRoot_991.w_10_0 ctx)
  h11_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_10_0 ctx) (parallel_generatedRoot_991.w_11_0 ctx)
  h14_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_991.w_14_0 ctx)
  h15_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_14_0 ctx) (parallel_generatedRoot_991.w_15_0 ctx)
  h16_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_15_0 ctx) (parallel_generatedRoot_991.w_16_0 ctx)
  h21_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_20_0 ctx) (parallel_generatedRoot_991.w_21_0 ctx)
  h24_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_23_0 ctx) (parallel_generatedRoot_991.w_24_0 ctx)
  h25_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_24_0 ctx) (parallel_generatedRoot_991.w_25_0 ctx)
  t10 : 0 ≤ (0)
  t11 : (0) < 25
  t12 : 0 ≤ 0
  t13 : 0 < 1
  t14 : 1 ≤ 1
  t15 : 0 ≤ (625 * (Int.ofNat i_0))
  t16 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t17 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h30_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_29_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_991.w_30_0 ctx)
  h32_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_32_0 ctx)
  h33_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_32_0 ctx) (parallel_generatedRoot_991.w_33_0 ctx)
  h36_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_35_0 ctx) (parallel_generatedRoot_991.w_36_0 ctx)
  h37_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_36_0 ctx) (parallel_generatedRoot_991.w_37_0 ctx)
  t23 : 0 ≤ 0
  t24 : 0 < 1
  t25 : 1 ≤ 1
  t26 : 0 ≤ 0
  t27 : 0 < 25
  t28 : 25 ≤ 625
  h39_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_38_0 ctx) 0 1 0 25 (parallel_generatedRoot_991.w_39_0 ctx)
  h41_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_40_0 ctx) (parallel_generatedRoot_991.w_41_0 ctx)
  h42_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_41_0 ctx) (parallel_generatedRoot_991.w_42_0 ctx)

theorem parallel_generatedRoot_991.facts_0  {ctx : parallel_generatedRoot_991Ctx} {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.constraints_0 backend i_0 (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_29_0 ctx) (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_2_0 ctx) (parallel_generatedRoot_991.w_3_0 ctx) (parallel_generatedRoot_991.w_6_0 ctx) (parallel_generatedRoot_991.w_9_0 ctx) (parallel_generatedRoot_991.w_10_0 ctx) (parallel_generatedRoot_991.w_11_0 ctx) (parallel_generatedRoot_991.w_14_0 ctx) (parallel_generatedRoot_991.w_15_0 ctx) (parallel_generatedRoot_991.w_16_0 ctx) (parallel_generatedRoot_991.w_20_0 ctx) (parallel_generatedRoot_991.w_21_0 ctx) (parallel_generatedRoot_991.w_23_0 ctx) (parallel_generatedRoot_991.w_24_0 ctx) (parallel_generatedRoot_991.w_25_0 ctx) (parallel_generatedRoot_991.w_30_0 ctx) (parallel_generatedRoot_991.w_32_0 ctx) (parallel_generatedRoot_991.w_33_0 ctx) (parallel_generatedRoot_991.w_35_0 ctx) (parallel_generatedRoot_991.w_36_0 ctx) (parallel_generatedRoot_991.w_37_0 ctx) (parallel_generatedRoot_991.w_38_0 ctx) (parallel_generatedRoot_991.w_39_0 ctx) (parallel_generatedRoot_991.w_40_0 ctx) (parallel_generatedRoot_991.w_41_0 ctx) (parallel_generatedRoot_991.w_42_0 ctx)) :
    parallel_generatedRoot_991.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_991.Facts_1  (ctx : parallel_generatedRoot_991Ctx) : Prop where
  h44_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_991.w_28_0 ctx) (parallel_generatedRoot_991.w_43_0 ctx) (parallel_generatedRoot_991.w_44_0 ctx)
  t33 : 0 ≤ (24)
  h45_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_991_45 backend hashModel params i_0 i (current, ((parallel_generatedRoot_991.w_38_0 ctx), ((parallel_generatedRoot_991.w_42_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_991.w_44_0 ctx) (parallel_generatedRoot_991.w_45_0 ctx)
  t35 : 0 ≤ (1)
  h52_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_991_52 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_991.w_48_0 ctx), ((parallel_generatedRoot_991.w_49_0 ctx), ((parallel_generatedRoot_991.w_50_0 ctx), ((parallel_generatedRoot_991.w_51_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_991.w_47_0 ctx), ((parallel_generatedRoot_991.w_47_0 ctx), ((parallel_generatedRoot_991.w_47_0 ctx), ()))) ((parallel_generatedRoot_991.w_52_0 ctx), ((parallel_generatedRoot_991.w_52_1 ctx), ((parallel_generatedRoot_991.w_52_2 ctx), ())))
  h57_0 : MxxRuntime.modulusSwitchRuns (parallel_generatedRoot_991.w_56_0 ctx) (parallel_generatedRoot_991.w_57_0 ctx)
  h58_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_57_0 ctx) (parallel_generatedRoot_991.w_58_0 ctx)
  h63_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_62_0 ctx) (parallel_generatedRoot_991.w_63_0 ctx)
  h64_0 : MxxRuntime.concatColumns (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_64_0 ctx)
  h67_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_66_0 ctx) (parallel_generatedRoot_991.w_67_0 ctx)
  h68_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_67_0 ctx) (parallel_generatedRoot_991.w_68_0 ctx)
  h70_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_991.w_70_0 ctx)
  h71_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_70_0 ctx) (parallel_generatedRoot_991.w_71_0 ctx)
  h72_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_71_0 ctx) (parallel_generatedRoot_991.w_72_0 ctx)
  h77_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_76_0 ctx) (parallel_generatedRoot_991.w_77_0 ctx)
  h80_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_79_0 ctx) (parallel_generatedRoot_991.w_80_0 ctx)
  h81_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_80_0 ctx) (parallel_generatedRoot_991.w_81_0 ctx)
  t49 : 0 ≤ (0)
  t50 : (0) < 25
  t51 : 0 ≤ 0
  t52 : 0 < 1
  t53 : 1 ≤ 1
  t54 : 0 ≤ (625 * (Int.ofNat i_0))
  t55 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t56 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h85_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_29_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_991.w_85_0 ctx)
  h86_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_86_0 ctx)
  h87_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_86_0 ctx) (parallel_generatedRoot_991.w_87_0 ctx)
  h90_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_89_0 ctx) (parallel_generatedRoot_991.w_90_0 ctx)
  h91_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_90_0 ctx) (parallel_generatedRoot_991.w_91_0 ctx)
  t62 : 0 ≤ 0
  t63 : 0 < 1

theorem parallel_generatedRoot_991.facts_1  {ctx : parallel_generatedRoot_991Ctx} {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.constraints_1 backend hashModel params i_0 (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_29_0 ctx) (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_48_0 ctx) (parallel_generatedRoot_991.w_49_0 ctx) (parallel_generatedRoot_991.w_50_0 ctx) (parallel_generatedRoot_991.w_51_0 ctx) (parallel_generatedRoot_991.w_28_0 ctx) (parallel_generatedRoot_991.w_38_0 ctx) (parallel_generatedRoot_991.w_42_0 ctx) (parallel_generatedRoot_991.w_43_0 ctx) (parallel_generatedRoot_991.w_44_0 ctx) (parallel_generatedRoot_991.w_45_0 ctx) (parallel_generatedRoot_991.w_47_0 ctx) (parallel_generatedRoot_991.w_52_0 ctx) (parallel_generatedRoot_991.w_52_1 ctx) (parallel_generatedRoot_991.w_52_2 ctx) (parallel_generatedRoot_991.w_56_0 ctx) (parallel_generatedRoot_991.w_57_0 ctx) (parallel_generatedRoot_991.w_58_0 ctx) (parallel_generatedRoot_991.w_62_0 ctx) (parallel_generatedRoot_991.w_63_0 ctx) (parallel_generatedRoot_991.w_64_0 ctx) (parallel_generatedRoot_991.w_66_0 ctx) (parallel_generatedRoot_991.w_67_0 ctx) (parallel_generatedRoot_991.w_68_0 ctx) (parallel_generatedRoot_991.w_70_0 ctx) (parallel_generatedRoot_991.w_71_0 ctx) (parallel_generatedRoot_991.w_72_0 ctx) (parallel_generatedRoot_991.w_76_0 ctx) (parallel_generatedRoot_991.w_77_0 ctx) (parallel_generatedRoot_991.w_79_0 ctx) (parallel_generatedRoot_991.w_80_0 ctx) (parallel_generatedRoot_991.w_81_0 ctx) (parallel_generatedRoot_991.w_85_0 ctx) (parallel_generatedRoot_991.w_86_0 ctx) (parallel_generatedRoot_991.w_87_0 ctx) (parallel_generatedRoot_991.w_89_0 ctx) (parallel_generatedRoot_991.w_90_0 ctx) (parallel_generatedRoot_991.w_91_0 ctx)) :
    parallel_generatedRoot_991.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_991.Facts_2  (ctx : parallel_generatedRoot_991Ctx) : Prop where
  t64 : 1 ≤ 1
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 625
  h93_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_92_0 ctx) 0 1 0 25 (parallel_generatedRoot_991.w_93_0 ctx)
  h95_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_94_0 ctx) (parallel_generatedRoot_991.w_95_0 ctx)
  h96_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_95_0 ctx) (parallel_generatedRoot_991.w_96_0 ctx)
  h98_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_991.w_84_0 ctx) (parallel_generatedRoot_991.w_97_0 ctx) (parallel_generatedRoot_991.w_98_0 ctx)
  t72 : 0 ≤ (24)
  h99_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_991_99 backend hashModel params i_0 i (current, ((parallel_generatedRoot_991.w_92_0 ctx), ((parallel_generatedRoot_991.w_96_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_991.w_98_0 ctx) (parallel_generatedRoot_991.w_99_0 ctx)
  t74 : 0 ≤ (1)
  h102_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_81 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_991.w_48_0 ctx), ((parallel_generatedRoot_991.w_49_0 ctx), ((parallel_generatedRoot_991.w_50_0 ctx), ((parallel_generatedRoot_991.w_51_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_991.w_101_0 ctx), ((parallel_generatedRoot_991.w_101_0 ctx), ((parallel_generatedRoot_991.w_101_0 ctx), ()))) ((parallel_generatedRoot_991.w_102_0 ctx), ((parallel_generatedRoot_991.w_102_1 ctx), ((parallel_generatedRoot_991.w_102_2 ctx), ())))
  h106_0 : MxxRuntime.modulusSwitchRuns (parallel_generatedRoot_991.w_105_0 ctx) (parallel_generatedRoot_991.w_106_0 ctx)
  h107_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_106_0 ctx) (parallel_generatedRoot_991.w_107_0 ctx)
  h113_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_112_0 ctx) (parallel_generatedRoot_991.w_113_0 ctx)
  h114_0 : MxxRuntime.concatColumns (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_114_0 ctx)
  h117_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_116_0 ctx) (parallel_generatedRoot_991.w_117_0 ctx)
  h118_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_117_0 ctx) (parallel_generatedRoot_991.w_118_0 ctx)
  h120_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_991.w_120_0 ctx)
  h121_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_120_0 ctx) (parallel_generatedRoot_991.w_121_0 ctx)
  h122_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_121_0 ctx) (parallel_generatedRoot_991.w_122_0 ctx)
  h127_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_126_0 ctx) (parallel_generatedRoot_991.w_127_0 ctx)
  h130_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_129_0 ctx) (parallel_generatedRoot_991.w_130_0 ctx)
  h131_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_130_0 ctx) (parallel_generatedRoot_991.w_131_0 ctx)
  t88 : 0 ≤ (0)
  t89 : (0) < 25
  t90 : 0 ≤ 0
  t91 : 0 < 1
  t92 : 1 ≤ 1
  t93 : 0 ≤ (625 * (Int.ofNat i_0))
  t94 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t95 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500

theorem parallel_generatedRoot_991.facts_2  {ctx : parallel_generatedRoot_991Ctx} {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.constraints_2 backend hashModel params i_0 (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_48_0 ctx) (parallel_generatedRoot_991.w_49_0 ctx) (parallel_generatedRoot_991.w_50_0 ctx) (parallel_generatedRoot_991.w_51_0 ctx) (parallel_generatedRoot_991.w_84_0 ctx) (parallel_generatedRoot_991.w_92_0 ctx) (parallel_generatedRoot_991.w_93_0 ctx) (parallel_generatedRoot_991.w_94_0 ctx) (parallel_generatedRoot_991.w_95_0 ctx) (parallel_generatedRoot_991.w_96_0 ctx) (parallel_generatedRoot_991.w_97_0 ctx) (parallel_generatedRoot_991.w_98_0 ctx) (parallel_generatedRoot_991.w_99_0 ctx) (parallel_generatedRoot_991.w_101_0 ctx) (parallel_generatedRoot_991.w_102_0 ctx) (parallel_generatedRoot_991.w_102_1 ctx) (parallel_generatedRoot_991.w_102_2 ctx) (parallel_generatedRoot_991.w_105_0 ctx) (parallel_generatedRoot_991.w_106_0 ctx) (parallel_generatedRoot_991.w_107_0 ctx) (parallel_generatedRoot_991.w_112_0 ctx) (parallel_generatedRoot_991.w_113_0 ctx) (parallel_generatedRoot_991.w_114_0 ctx) (parallel_generatedRoot_991.w_116_0 ctx) (parallel_generatedRoot_991.w_117_0 ctx) (parallel_generatedRoot_991.w_118_0 ctx) (parallel_generatedRoot_991.w_120_0 ctx) (parallel_generatedRoot_991.w_121_0 ctx) (parallel_generatedRoot_991.w_122_0 ctx) (parallel_generatedRoot_991.w_126_0 ctx) (parallel_generatedRoot_991.w_127_0 ctx) (parallel_generatedRoot_991.w_129_0 ctx) (parallel_generatedRoot_991.w_130_0 ctx) (parallel_generatedRoot_991.w_131_0 ctx)) :
    parallel_generatedRoot_991.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_991.Facts_3  (ctx : parallel_generatedRoot_991Ctx) : Prop where
  h135_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_29_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_991.w_135_0 ctx)
  h136_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_136_0 ctx)
  h137_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_136_0 ctx) (parallel_generatedRoot_991.w_137_0 ctx)
  h140_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_139_0 ctx) (parallel_generatedRoot_991.w_140_0 ctx)
  h141_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_140_0 ctx) (parallel_generatedRoot_991.w_141_0 ctx)
  t101 : 0 ≤ 0
  t102 : 0 < 1
  t103 : 1 ≤ 1
  t104 : 0 ≤ 0
  t105 : 0 < 25
  t106 : 25 ≤ 625
  h143_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_142_0 ctx) 0 1 0 25 (parallel_generatedRoot_991.w_143_0 ctx)
  h145_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_144_0 ctx) (parallel_generatedRoot_991.w_145_0 ctx)
  h146_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_145_0 ctx) (parallel_generatedRoot_991.w_146_0 ctx)
  h148_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_991.w_134_0 ctx) (parallel_generatedRoot_991.w_147_0 ctx) (parallel_generatedRoot_991.w_148_0 ctx)
  t111 : 0 ≤ (24)
  h149_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => scope_82 backend hashModel params i_0 i (current, ((parallel_generatedRoot_991.w_142_0 ctx), ((parallel_generatedRoot_991.w_146_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_991.w_148_0 ctx) (parallel_generatedRoot_991.w_149_0 ctx)
  t113 : 0 ≤ (1)
  h152_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_83 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_991.w_48_0 ctx), ((parallel_generatedRoot_991.w_49_0 ctx), ((parallel_generatedRoot_991.w_50_0 ctx), ((parallel_generatedRoot_991.w_51_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_991.w_151_0 ctx), ((parallel_generatedRoot_991.w_151_0 ctx), ((parallel_generatedRoot_991.w_151_0 ctx), ()))) ((parallel_generatedRoot_991.w_152_0 ctx), ((parallel_generatedRoot_991.w_152_1 ctx), ((parallel_generatedRoot_991.w_152_2 ctx), ())))
  h156_0 : MxxRuntime.modulusSwitchRuns (parallel_generatedRoot_991.w_155_0 ctx) (parallel_generatedRoot_991.w_156_0 ctx)
  h157_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_156_0 ctx) (parallel_generatedRoot_991.w_157_0 ctx)
  h163_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_162_0 ctx) (parallel_generatedRoot_991.w_163_0 ctx)
  h164_0 : MxxRuntime.concatColumns (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_164_0 ctx)
  h167_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_166_0 ctx) (parallel_generatedRoot_991.w_167_0 ctx)
  h168_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_167_0 ctx) (parallel_generatedRoot_991.w_168_0 ctx)
  h170_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_991.w_170_0 ctx)
  h171_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_170_0 ctx) (parallel_generatedRoot_991.w_171_0 ctx)
  h172_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_171_0 ctx) (parallel_generatedRoot_991.w_172_0 ctx)
  h177_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_176_0 ctx) (parallel_generatedRoot_991.w_177_0 ctx)
  h180_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_179_0 ctx) (parallel_generatedRoot_991.w_180_0 ctx)
  h181_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_180_0 ctx) (parallel_generatedRoot_991.w_181_0 ctx)
  t127 : 0 ≤ (0)

theorem parallel_generatedRoot_991.facts_3  {ctx : parallel_generatedRoot_991Ctx} {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.constraints_3 backend hashModel params i_0 (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_29_0 ctx) (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_48_0 ctx) (parallel_generatedRoot_991.w_49_0 ctx) (parallel_generatedRoot_991.w_50_0 ctx) (parallel_generatedRoot_991.w_51_0 ctx) (parallel_generatedRoot_991.w_134_0 ctx) (parallel_generatedRoot_991.w_135_0 ctx) (parallel_generatedRoot_991.w_136_0 ctx) (parallel_generatedRoot_991.w_137_0 ctx) (parallel_generatedRoot_991.w_139_0 ctx) (parallel_generatedRoot_991.w_140_0 ctx) (parallel_generatedRoot_991.w_141_0 ctx) (parallel_generatedRoot_991.w_142_0 ctx) (parallel_generatedRoot_991.w_143_0 ctx) (parallel_generatedRoot_991.w_144_0 ctx) (parallel_generatedRoot_991.w_145_0 ctx) (parallel_generatedRoot_991.w_146_0 ctx) (parallel_generatedRoot_991.w_147_0 ctx) (parallel_generatedRoot_991.w_148_0 ctx) (parallel_generatedRoot_991.w_149_0 ctx) (parallel_generatedRoot_991.w_151_0 ctx) (parallel_generatedRoot_991.w_152_0 ctx) (parallel_generatedRoot_991.w_152_1 ctx) (parallel_generatedRoot_991.w_152_2 ctx) (parallel_generatedRoot_991.w_155_0 ctx) (parallel_generatedRoot_991.w_156_0 ctx) (parallel_generatedRoot_991.w_157_0 ctx) (parallel_generatedRoot_991.w_162_0 ctx) (parallel_generatedRoot_991.w_163_0 ctx) (parallel_generatedRoot_991.w_164_0 ctx) (parallel_generatedRoot_991.w_166_0 ctx) (parallel_generatedRoot_991.w_167_0 ctx) (parallel_generatedRoot_991.w_168_0 ctx) (parallel_generatedRoot_991.w_170_0 ctx) (parallel_generatedRoot_991.w_171_0 ctx) (parallel_generatedRoot_991.w_172_0 ctx) (parallel_generatedRoot_991.w_176_0 ctx) (parallel_generatedRoot_991.w_177_0 ctx) (parallel_generatedRoot_991.w_179_0 ctx) (parallel_generatedRoot_991.w_180_0 ctx) (parallel_generatedRoot_991.w_181_0 ctx)) :
    parallel_generatedRoot_991.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_991.Facts_4  (ctx : parallel_generatedRoot_991Ctx) : Prop where
  t128 : (0) < 25
  t129 : 0 ≤ 0
  t130 : 0 < 1
  t131 : 1 ≤ 1
  t132 : 0 ≤ (625 * (Int.ofNat i_0))
  t133 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t134 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h185_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_29_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_991.w_185_0 ctx)
  h186_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_186_0 ctx)
  h187_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_186_0 ctx) (parallel_generatedRoot_991.w_187_0 ctx)
  h190_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_189_0 ctx) (parallel_generatedRoot_991.w_190_0 ctx)
  h191_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_190_0 ctx) (parallel_generatedRoot_991.w_191_0 ctx)
  t140 : 0 ≤ 0
  t141 : 0 < 1
  t142 : 1 ≤ 1
  t143 : 0 ≤ 0
  t144 : 0 < 25
  t145 : 25 ≤ 625
  h193_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_192_0 ctx) 0 1 0 25 (parallel_generatedRoot_991.w_193_0 ctx)
  h195_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_194_0 ctx) (parallel_generatedRoot_991.w_195_0 ctx)
  h196_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_195_0 ctx) (parallel_generatedRoot_991.w_196_0 ctx)
  h198_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_991.w_184_0 ctx) (parallel_generatedRoot_991.w_197_0 ctx) (parallel_generatedRoot_991.w_198_0 ctx)
  t150 : 0 ≤ (24)
  h199_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => scope_84 backend hashModel params i_0 i (current, ((parallel_generatedRoot_991.w_192_0 ctx), ((parallel_generatedRoot_991.w_196_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_991.w_198_0 ctx) (parallel_generatedRoot_991.w_199_0 ctx)
  t152 : 0 ≤ (1)
  h202_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_85 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_991.w_48_0 ctx), ((parallel_generatedRoot_991.w_49_0 ctx), ((parallel_generatedRoot_991.w_50_0 ctx), ((parallel_generatedRoot_991.w_51_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_991.w_201_0 ctx), ((parallel_generatedRoot_991.w_201_0 ctx), ((parallel_generatedRoot_991.w_201_0 ctx), ()))) ((parallel_generatedRoot_991.w_202_0 ctx), ((parallel_generatedRoot_991.w_202_1 ctx), ((parallel_generatedRoot_991.w_202_2 ctx), ())))
  h206_0 : MxxRuntime.modulusSwitchRuns (parallel_generatedRoot_991.w_205_0 ctx) (parallel_generatedRoot_991.w_206_0 ctx)
  h207_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_206_0 ctx) (parallel_generatedRoot_991.w_207_0 ctx)
  h213_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_212_0 ctx) (parallel_generatedRoot_991.w_213_0 ctx)
  h214_0 : MxxRuntime.concatColumns (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_214_0 ctx)
  h217_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_216_0 ctx) (parallel_generatedRoot_991.w_217_0 ctx)
  h218_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_217_0 ctx) (parallel_generatedRoot_991.w_218_0 ctx)

theorem parallel_generatedRoot_991.facts_4  {ctx : parallel_generatedRoot_991Ctx} {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.constraints_4 backend hashModel params i_0 (parallel_generatedRoot_991.w_4_0 ctx) (parallel_generatedRoot_991.w_5_0 ctx) (parallel_generatedRoot_991.w_29_0 ctx) (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_48_0 ctx) (parallel_generatedRoot_991.w_49_0 ctx) (parallel_generatedRoot_991.w_50_0 ctx) (parallel_generatedRoot_991.w_51_0 ctx) (parallel_generatedRoot_991.w_184_0 ctx) (parallel_generatedRoot_991.w_185_0 ctx) (parallel_generatedRoot_991.w_186_0 ctx) (parallel_generatedRoot_991.w_187_0 ctx) (parallel_generatedRoot_991.w_189_0 ctx) (parallel_generatedRoot_991.w_190_0 ctx) (parallel_generatedRoot_991.w_191_0 ctx) (parallel_generatedRoot_991.w_192_0 ctx) (parallel_generatedRoot_991.w_193_0 ctx) (parallel_generatedRoot_991.w_194_0 ctx) (parallel_generatedRoot_991.w_195_0 ctx) (parallel_generatedRoot_991.w_196_0 ctx) (parallel_generatedRoot_991.w_197_0 ctx) (parallel_generatedRoot_991.w_198_0 ctx) (parallel_generatedRoot_991.w_199_0 ctx) (parallel_generatedRoot_991.w_201_0 ctx) (parallel_generatedRoot_991.w_202_0 ctx) (parallel_generatedRoot_991.w_202_1 ctx) (parallel_generatedRoot_991.w_202_2 ctx) (parallel_generatedRoot_991.w_205_0 ctx) (parallel_generatedRoot_991.w_206_0 ctx) (parallel_generatedRoot_991.w_207_0 ctx) (parallel_generatedRoot_991.w_212_0 ctx) (parallel_generatedRoot_991.w_213_0 ctx) (parallel_generatedRoot_991.w_214_0 ctx) (parallel_generatedRoot_991.w_216_0 ctx) (parallel_generatedRoot_991.w_217_0 ctx) (parallel_generatedRoot_991.w_218_0 ctx)) :
    parallel_generatedRoot_991.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_991.Facts_5  (ctx : parallel_generatedRoot_991Ctx) : Prop where
  h220_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_991.w_220_0 ctx)
  h221_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_220_0 ctx) (parallel_generatedRoot_991.w_221_0 ctx)
  h222_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_991.w_221_0 ctx) (parallel_generatedRoot_991.w_222_0 ctx)
  h227_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_226_0 ctx) (parallel_generatedRoot_991.w_227_0 ctx)
  h230_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_229_0 ctx) (parallel_generatedRoot_991.w_230_0 ctx)
  h231_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_230_0 ctx) (parallel_generatedRoot_991.w_231_0 ctx)
  t166 : 0 ≤ (0)
  t167 : (0) < 25
  t168 : 0 ≤ 0
  t169 : 0 < 1
  t170 : 1 ≤ 1
  t171 : 0 ≤ (625 * (Int.ofNat i_0))
  t172 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t173 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h235_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_29_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_991.w_235_0 ctx)
  h236_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_236_0 ctx)
  h237_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_236_0 ctx) (parallel_generatedRoot_991.w_237_0 ctx)
  h240_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_239_0 ctx) (parallel_generatedRoot_991.w_240_0 ctx)
  h241_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_240_0 ctx) (parallel_generatedRoot_991.w_241_0 ctx)
  t179 : 0 ≤ 0
  t180 : 0 < 1
  t181 : 1 ≤ 1
  t182 : 0 ≤ 0
  t183 : 0 < 25
  t184 : 25 ≤ 625
  h243_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_991.w_242_0 ctx) 0 1 0 25 (parallel_generatedRoot_991.w_243_0 ctx)
  h245_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_991.w_244_0 ctx) (parallel_generatedRoot_991.w_245_0 ctx)
  h246_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_991.w_245_0 ctx) (parallel_generatedRoot_991.w_246_0 ctx)
  h248_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_991.w_234_0 ctx) (parallel_generatedRoot_991.w_247_0 ctx) (parallel_generatedRoot_991.w_248_0 ctx)
  t189 : 0 ≤ (24)
  h249_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => scope_86 backend hashModel params i_0 i (current, ((parallel_generatedRoot_991.w_242_0 ctx), ((parallel_generatedRoot_991.w_246_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_991.w_248_0 ctx) (parallel_generatedRoot_991.w_249_0 ctx)
  t191 : 0 ≤ (1)

theorem parallel_generatedRoot_991.facts_5  {ctx : parallel_generatedRoot_991Ctx} {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.constraints_5 backend hashModel params i_0 (parallel_generatedRoot_991.w_29_0 ctx) (parallel_generatedRoot_991.w_31_0 ctx) (parallel_generatedRoot_991.w_220_0 ctx) (parallel_generatedRoot_991.w_221_0 ctx) (parallel_generatedRoot_991.w_222_0 ctx) (parallel_generatedRoot_991.w_226_0 ctx) (parallel_generatedRoot_991.w_227_0 ctx) (parallel_generatedRoot_991.w_229_0 ctx) (parallel_generatedRoot_991.w_230_0 ctx) (parallel_generatedRoot_991.w_231_0 ctx) (parallel_generatedRoot_991.w_234_0 ctx) (parallel_generatedRoot_991.w_235_0 ctx) (parallel_generatedRoot_991.w_236_0 ctx) (parallel_generatedRoot_991.w_237_0 ctx) (parallel_generatedRoot_991.w_239_0 ctx) (parallel_generatedRoot_991.w_240_0 ctx) (parallel_generatedRoot_991.w_241_0 ctx) (parallel_generatedRoot_991.w_242_0 ctx) (parallel_generatedRoot_991.w_243_0 ctx) (parallel_generatedRoot_991.w_244_0 ctx) (parallel_generatedRoot_991.w_245_0 ctx) (parallel_generatedRoot_991.w_246_0 ctx) (parallel_generatedRoot_991.w_247_0 ctx) (parallel_generatedRoot_991.w_248_0 ctx) (parallel_generatedRoot_991.w_249_0 ctx)) :
    parallel_generatedRoot_991.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_991.Facts_6  (ctx : parallel_generatedRoot_991Ctx) (outputs : mxxMat0) : Prop where
  h252_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_87 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_991.w_48_0 ctx), ((parallel_generatedRoot_991.w_49_0 ctx), ((parallel_generatedRoot_991.w_50_0 ctx), ((parallel_generatedRoot_991.w_51_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_991.w_251_0 ctx), ((parallel_generatedRoot_991.w_251_0 ctx), ((parallel_generatedRoot_991.w_251_0 ctx), ()))) ((parallel_generatedRoot_991.w_252_0 ctx), ((parallel_generatedRoot_991.w_252_1 ctx), ((parallel_generatedRoot_991.w_252_2 ctx), ())))
  h256_0 : MxxRuntime.modulusSwitchRuns (parallel_generatedRoot_991.w_255_0 ctx) (parallel_generatedRoot_991.w_256_0 ctx)
  h257_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_991.w_256_0 ctx) (parallel_generatedRoot_991.w_257_0 ctx)
  hout : outputs = parallel_generatedRoot_991.outs ctx

theorem parallel_generatedRoot_991.facts_6  {ctx : parallel_generatedRoot_991Ctx} {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.constraints_6 backend hashModel params i_0 (parallel_generatedRoot_991.w_48_0 ctx) (parallel_generatedRoot_991.w_49_0 ctx) (parallel_generatedRoot_991.w_50_0 ctx) (parallel_generatedRoot_991.w_51_0 ctx) (parallel_generatedRoot_991.w_251_0 ctx) (parallel_generatedRoot_991.w_252_0 ctx) (parallel_generatedRoot_991.w_252_1 ctx) (parallel_generatedRoot_991.w_252_2 ctx) (parallel_generatedRoot_991.w_255_0 ctx) (parallel_generatedRoot_991.w_256_0 ctx) (parallel_generatedRoot_991.w_257_0 ctx) (parallel_generatedRoot_991.w_260_0 ctx) outputs) :
    parallel_generatedRoot_991.Facts_6 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩

structure parallel_generatedRoot_991.Facts  (ctx : parallel_generatedRoot_991Ctx) (outputs : mxxMat0) : Prop where
  c0 : parallel_generatedRoot_991.Facts_0 ctx
  c1 : parallel_generatedRoot_991.Facts_1 ctx
  c2 : parallel_generatedRoot_991.Facts_2 ctx
  c3 : parallel_generatedRoot_991.Facts_3 ctx
  c4 : parallel_generatedRoot_991.Facts_4 ctx
  c5 : parallel_generatedRoot_991.Facts_5 ctx
  c6 : parallel_generatedRoot_991.Facts_6 ctx outputs

theorem parallel_generatedRoot_991.facts  (ctx : parallel_generatedRoot_991Ctx) {outputs : mxxMat0}
    (h : parallel_generatedRoot_991.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_991.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_991.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_991.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_991.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_991.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_991.facts_4 (ctx := ctx) (outputs := outputs) t4.1
  have t5 := t4.2
  have r5 := parallel_generatedRoot_991.facts_5 (ctx := ctx) (outputs := outputs) t5.1
  have t6 := t5.2
  have r6 := parallel_generatedRoot_991.facts_6 (ctx := ctx) (outputs := outputs) t6
  ⟨r0, r1, r2, r3, r4, r5, r6⟩

end

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel params)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  h19_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_19_0 ctx)
  h20_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_20_0 ctx)
  h23_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_23_0 ctx)
  h24_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_24_0 ctx)
  t4 : (16) = 16
  h29_5 : (∀ i : Fin 16, ∃ w_16_0_gathered : mxxMat0, ∃ w_17_0_gathered : mxxMat0, ∃ w_22_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_18_0 ctx) i) ∧ ((generatedRoot.w_18_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_18_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_18_0 ctx) i) ∧ ((generatedRoot.w_18_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_17_0 ctx) ((generatedRoot.w_18_0 ctx) i) w_17_0_gathered ∧ 0 ≤ ((generatedRoot.w_18_0 ctx) i) ∧ ((generatedRoot.w_18_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_22_0 ctx) ((generatedRoot.w_18_0 ctx) i) w_22_0_gathered ∧ parallel_generatedRoot_29 backend hashModel params i (((generatedRoot.w_18_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_7_0 ctx), (w_17_0_gathered, ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_21_0 ctx), (w_22_0_gathered, ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_24_0 ctx), ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_26_0 ctx), (((generatedRoot.w_27_0 ctx) i), (((generatedRoot.w_28_0 ctx) i), ())))))))))))))) (((generatedRoot.w_29_0 ctx) i), (((generatedRoot.w_29_1 ctx) i), (((generatedRoot.w_29_2 ctx) i), (((generatedRoot.w_29_3 ctx) i), (((generatedRoot.w_29_4 ctx) i), (((generatedRoot.w_29_5 ctx) i), ())))))))
  t6 : (8) = 8
  h30_5 : (∀ i : Fin 8, parallel_generatedRoot_30 backend hashModel params i ((generatedRoot.w_29_0 ctx), ((generatedRoot.w_29_1 ctx), ((generatedRoot.w_29_2 ctx), ((generatedRoot.w_29_3 ctx), ((generatedRoot.w_29_4 ctx), ((generatedRoot.w_29_5 ctx), ())))))) (((generatedRoot.w_30_0 ctx) i), (((generatedRoot.w_30_1 ctx) i), (((generatedRoot.w_30_2 ctx) i), (((generatedRoot.w_30_3 ctx) i), (((generatedRoot.w_30_4 ctx) i), (((generatedRoot.w_30_5 ctx) i), ())))))))
  t8 : (16) = 16
  h33_5 : (∀ i : Fin 16, ∃ w_16_0_gathered : mxxMat0, ∃ w_17_0_gathered : mxxMat0, ∃ w_22_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_15_0 ctx) i) ∧ ((generatedRoot.w_15_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_15_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_15_0 ctx) i) ∧ ((generatedRoot.w_15_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_17_0 ctx) ((generatedRoot.w_15_0 ctx) i) w_17_0_gathered ∧ 0 ≤ ((generatedRoot.w_15_0 ctx) i) ∧ ((generatedRoot.w_15_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_22_0 ctx) ((generatedRoot.w_15_0 ctx) i) w_22_0_gathered ∧ parallel_generatedRoot_33 backend hashModel params i (((generatedRoot.w_15_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_7_0 ctx), (w_17_0_gathered, ((generatedRoot.w_30_0 ctx), ((generatedRoot.w_30_2 ctx), ((generatedRoot.w_30_4 ctx), (w_22_0_gathered, ((generatedRoot.w_30_1 ctx), ((generatedRoot.w_30_3 ctx), ((generatedRoot.w_30_5 ctx), ((generatedRoot.w_26_0 ctx), (((generatedRoot.w_31_0 ctx) i), (((generatedRoot.w_32_0 ctx) i), ())))))))))))))) (((generatedRoot.w_33_0 ctx) i), (((generatedRoot.w_33_1 ctx) i), (((generatedRoot.w_33_2 ctx) i), (((generatedRoot.w_33_3 ctx) i), (((generatedRoot.w_33_4 ctx) i), (((generatedRoot.w_33_5 ctx) i), ())))))))
  t10 : (8) = 8
  h34_5 : (∀ i : Fin 8, parallel_generatedRoot_34 backend hashModel params i ((generatedRoot.w_33_0 ctx), ((generatedRoot.w_33_1 ctx), ((generatedRoot.w_33_2 ctx), ((generatedRoot.w_33_3 ctx), ((generatedRoot.w_33_4 ctx), ((generatedRoot.w_33_5 ctx), ())))))) (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_1 ctx) i), (((generatedRoot.w_34_2 ctx) i), (((generatedRoot.w_34_3 ctx) i), (((generatedRoot.w_34_4 ctx) i), (((generatedRoot.w_34_5 ctx) i), ())))))))
  t12 : (8) = 8
  h35_5 : (∀ i : Fin 8, parallel_generatedRoot_35 backend hashModel params i ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_14_0 ctx), (((generatedRoot.w_34_4 ctx) i), (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_2 ctx) i), (((generatedRoot.w_34_5 ctx) i), (((generatedRoot.w_34_1 ctx) i), (((generatedRoot.w_34_3 ctx) i), ()))))))))))))) (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_35_1 ctx) i), (((generatedRoot.w_35_2 ctx) i), (((generatedRoot.w_35_3 ctx) i), (((generatedRoot.w_35_4 ctx) i), (((generatedRoot.w_35_5 ctx) i), ())))))))
  t14 : (8) = 8
  h36_5 : (∀ i : Fin 8, parallel_generatedRoot_36 backend hashModel params i ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), (((generatedRoot.w_35_4 ctx) i), (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_35_2 ctx) i), (((generatedRoot.w_35_5 ctx) i), (((generatedRoot.w_35_1 ctx) i), (((generatedRoot.w_35_3 ctx) i), ()))))))))))))) (((generatedRoot.w_36_0 ctx) i), (((generatedRoot.w_36_1 ctx) i), (((generatedRoot.w_36_2 ctx) i), (((generatedRoot.w_36_3 ctx) i), (((generatedRoot.w_36_4 ctx) i), (((generatedRoot.w_36_5 ctx) i), ())))))))
  t16 : (8) = 8
  h37_5 : (∀ i : Fin 8, parallel_generatedRoot_37 backend hashModel params i (((generatedRoot.w_36_0 ctx) i), (((generatedRoot.w_36_1 ctx) i), (((generatedRoot.w_36_2 ctx) i), (((generatedRoot.w_36_3 ctx) i), (((generatedRoot.w_36_4 ctx) i), (((generatedRoot.w_36_5 ctx) i), ())))))) (((generatedRoot.w_37_0 ctx) i), (((generatedRoot.w_37_1 ctx) i), (((generatedRoot.w_37_2 ctx) i), (((generatedRoot.w_37_3 ctx) i), (((generatedRoot.w_37_4 ctx) i), (((generatedRoot.w_37_5 ctx) i), ())))))))
  h38_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_0 ctx) (0) (generatedRoot.w_38_0 ctx)
  h39_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_0 ctx) (1) (generatedRoot.w_39_0 ctx)
  h41_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_1 ctx) (0) (generatedRoot.w_41_0 ctx)
  h42_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_1 ctx) (1) (generatedRoot.w_42_0 ctx)
  h44_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_2 ctx) (0) (generatedRoot.w_44_0 ctx)
  h45_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_2 ctx) (1) (generatedRoot.w_45_0 ctx)
  h47_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_3 ctx) (0) (generatedRoot.w_47_0 ctx)
  h48_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_3 ctx) (1) (generatedRoot.w_48_0 ctx)
  h50_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_4 ctx) (0) (generatedRoot.w_50_0 ctx)
  h51_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_4 ctx) (1) (generatedRoot.w_51_0 ctx)
  h53_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_5 ctx) (0) (generatedRoot.w_53_0 ctx)
  h54_0 : MxxRuntime.familyGetStatic (generatedRoot.w_37_5 ctx) (1) (generatedRoot.w_54_0 ctx)
  t30 : 0 ≤ (6)
  h56_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_56 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_37_0 ctx), ((generatedRoot.w_37_1 ctx), ((generatedRoot.w_37_2 ctx), ((generatedRoot.w_37_3 ctx), ((generatedRoot.w_37_4 ctx), ((generatedRoot.w_37_5 ctx), ())))))))))))) next) (Int.toNat (6)) ((generatedRoot.w_40_0 ctx), ((generatedRoot.w_43_0 ctx), ((generatedRoot.w_46_0 ctx), ((generatedRoot.w_49_0 ctx), ((generatedRoot.w_52_0 ctx), ((generatedRoot.w_55_0 ctx), ())))))) ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_5 ctx), ()))))))

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel params (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_29_1 ctx) (generatedRoot.w_29_2 ctx) (generatedRoot.w_29_3 ctx) (generatedRoot.w_29_4 ctx) (generatedRoot.w_29_5 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_30_1 ctx) (generatedRoot.w_30_2 ctx) (generatedRoot.w_30_3 ctx) (generatedRoot.w_30_4 ctx) (generatedRoot.w_30_5 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_33_4 ctx) (generatedRoot.w_33_5 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_34_1 ctx) (generatedRoot.w_34_2 ctx) (generatedRoot.w_34_3 ctx) (generatedRoot.w_34_4 ctx) (generatedRoot.w_34_5 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_35_1 ctx) (generatedRoot.w_35_2 ctx) (generatedRoot.w_35_3 ctx) (generatedRoot.w_35_4 ctx) (generatedRoot.w_35_5 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_36_1 ctx) (generatedRoot.w_36_2 ctx) (generatedRoot.w_36_3 ctx) (generatedRoot.w_36_4 ctx) (generatedRoot.w_36_5 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_37_1 ctx) (generatedRoot.w_37_2 ctx) (generatedRoot.w_37_3 ctx) (generatedRoot.w_37_4 ctx) (generatedRoot.w_37_5 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  h57_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_57_0 ctx)
  h58_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx)
  h61_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx)
  h62_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx)
  h64_0 : MxxRuntime.familyGetStatic (generatedRoot.w_17_0 ctx) (1) (generatedRoot.w_64_0 ctx)
  h65_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_65_0 ctx)
  h68_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx)
  h69_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_68_0 ctx) (generatedRoot.w_69_0 ctx)
  h72_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_72_0 ctx)
  h73_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx)
  h75_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx)
  h78_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx)
  h79_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx)
  h82_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_82_0 ctx)
  h85_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx)
  h86_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx)
  h91_0 : MxxRuntime.familyGetStatic (generatedRoot.w_90_0 ctx) (0) (generatedRoot.w_91_0 ctx)
  h93_0 : MxxRuntime.familyGetStatic (generatedRoot.w_92_0 ctx) (0) (generatedRoot.w_93_0 ctx)
  h95_0 : MxxRuntime.familyGetStatic (generatedRoot.w_94_0 ctx) (0) (generatedRoot.w_95_0 ctx)
  h97_0 : MxxRuntime.familyGetStatic (generatedRoot.w_96_0 ctx) (0) (generatedRoot.w_97_0 ctx)
  h101_0 : MxxRuntime.familyGetStatic (generatedRoot.w_100_0 ctx) (1) (generatedRoot.w_101_0 ctx)
  h102_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_102_0 ctx)
  h103_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_103_0 ctx)
  h102_0_1 : MxxRuntime.concatRows (generatedRoot.w_101_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_concat_1 ctx)
  h104_0 : MxxRuntime.concatRows (generatedRoot.w_104_concat_1 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx)
  h106_0 : MxxRuntime.familyGetStatic (generatedRoot.w_105_0 ctx) (1) (generatedRoot.w_106_0 ctx)
  h107_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_107_0 ctx)
  h108_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_108_0 ctx)
  h107_0_1 : MxxRuntime.concatRows (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_concat_1 ctx)
  h109_0 : MxxRuntime.concatRows (generatedRoot.w_109_concat_1 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx)
  h110_0 : MxxRuntime.concatColumns (generatedRoot.w_104_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx)
  h111_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_111_0 ctx)

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_7_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_104_concat_1 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_109_concat_1 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
  h112_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_112_0 ctx)
  h113_0 : MxxRuntime.concatRows (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx)
  h114_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx)
  h115_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx)
  t68 : 0 ≤ 0
  t69 : 0 < 1
  t70 : 1 ≤ 3
  t71 : 0 ≤ 0
  t72 : 0 < 25
  t73 : 25 ≤ 25
  h117_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 0 1 0 25 (generatedRoot.w_117_0 ctx)
  h120_0 : MxxRuntime.familyGetStatic (generatedRoot.w_119_0 ctx) (1) (generatedRoot.w_120_0 ctx)
  t76 : 0 ≤ 1
  t77 : 1 < 2
  t78 : 2 ≤ 3
  t79 : 0 ≤ 0
  t80 : 0 < 25
  t81 : 25 ≤ 25
  h123_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 1 2 0 25 (generatedRoot.w_123_0 ctx)
  t83 : 0 ≤ 2
  t84 : 2 < 3
  t85 : 3 ≤ 3
  t86 : 0 ≤ 0
  t87 : 0 < 25
  t88 : 25 ≤ 25
  h124_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 2 3 0 25 (generatedRoot.w_124_0 ctx)
  h123_0_1 : MxxRuntime.concatRows (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_concat_1 ctx)
  h125_0 : MxxRuntime.concatRows (generatedRoot.w_125_concat_1 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_125_0 ctx)
  h126_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_126_0 ctx)
  h127_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_127_0 ctx)
  h128_0 : MxxRuntime.concatRows (generatedRoot.w_126_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx)
  h129_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend hashModel (generatedRoot.w_7_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_125_concat_1 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
  h130_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx)
  t97 : 0 ≤ 0
  t98 : 0 < 1
  t99 : 1 ≤ 3
  t100 : 0 ≤ 0
  t101 : 0 < 25
  t102 : 25 ≤ 25
  h132_0 : MxxRuntime.sliceMatrix (generatedRoot.w_131_0 ctx) 0 1 0 25 (generatedRoot.w_132_0 ctx)
  t104 : 0 ≤ 1
  t105 : 1 < 2
  t106 : 2 ≤ 3
  t107 : 0 ≤ 0
  t108 : 0 < 25
  t109 : 25 ≤ 25
  h136_0 : MxxRuntime.sliceMatrix (generatedRoot.w_131_0 ctx) 1 2 0 25 (generatedRoot.w_136_0 ctx)
  t111 : 0 ≤ 2
  t112 : 2 < 3
  t113 : 3 ≤ 3
  t114 : 0 ≤ 0
  t115 : 0 < 25
  t116 : 25 ≤ 25
  h137_0 : MxxRuntime.sliceMatrix (generatedRoot.w_131_0 ctx) 2 3 0 25 (generatedRoot.w_137_0 ctx)
  h136_0_1 : MxxRuntime.concatRows (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_concat_1 ctx)
  h138_0 : MxxRuntime.concatRows (generatedRoot.w_138_concat_1 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_138_0 ctx)
  h139_0 : MxxRuntime.concatColumns (generatedRoot.w_125_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx)
  h140_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat5), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat6), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_26_0 ctx)) (generatedRoot.w_140_0 ctx)
  h142_0 : MxxRuntime.familyGetStatic (generatedRoot.w_141_0 ctx) (0) (generatedRoot.w_142_0 ctx)
  h143_0 : MxxRuntime.concatRows (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_143_0 ctx)
  h144_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx)
  h145_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx)
  t126 : 0 ≤ 0
  t127 : 0 < 1

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_3 backend hashModel (generatedRoot.w_26_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_138_concat_1 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4  (ctx : generatedRootCtx) : Prop where
  t128 : 1 ≤ 3
  t129 : 0 ≤ 0
  t130 : 0 < 25
  t131 : 25 ≤ 25
  h147_0 : MxxRuntime.sliceMatrix (generatedRoot.w_146_0 ctx) 0 1 0 25 (generatedRoot.w_147_0 ctx)
  t133 : 0 ≤ 1
  t134 : 1 < 2
  t135 : 2 ≤ 3
  t136 : 0 ≤ 0
  t137 : 0 < 25
  t138 : 25 ≤ 25
  h148_0 : MxxRuntime.sliceMatrix (generatedRoot.w_146_0 ctx) 1 2 0 25 (generatedRoot.w_148_0 ctx)
  t140 : 0 ≤ 2
  t141 : 2 < 3
  t142 : 3 ≤ 3
  t143 : 0 ≤ 0
  t144 : 0 < 25
  t145 : 25 ≤ 25
  h149_0 : MxxRuntime.sliceMatrix (generatedRoot.w_146_0 ctx) 2 3 0 25 (generatedRoot.w_149_0 ctx)
  h150_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat5), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat6), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_26_0 ctx)) (generatedRoot.w_150_0 ctx)
  h152_0 : MxxRuntime.familyGetStatic (generatedRoot.w_151_0 ctx) (0) (generatedRoot.w_152_0 ctx)
  h153_0 : MxxRuntime.concatRows (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_153_0 ctx)
  h154_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx)
  h155_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx)
  t152 : 0 ≤ 0
  t153 : 0 < 1
  t154 : 1 ≤ 3
  t155 : 0 ≤ 0
  t156 : 0 < 25
  t157 : 25 ≤ 25
  h157_0 : MxxRuntime.sliceMatrix (generatedRoot.w_156_0 ctx) 0 1 0 25 (generatedRoot.w_157_0 ctx)
  t159 : 0 ≤ 1

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_4 backend hashModel (generatedRoot.w_26_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
  t160 : 1 < 2
  t161 : 2 ≤ 3
  t162 : 0 ≤ 0
  t163 : 0 < 25
  t164 : 25 ≤ 25
  h158_0 : MxxRuntime.sliceMatrix (generatedRoot.w_156_0 ctx) 1 2 0 25 (generatedRoot.w_158_0 ctx)
  t166 : 0 ≤ 2
  t167 : 2 < 3
  t168 : 3 ≤ 3
  t169 : 0 ≤ 0
  t170 : 0 < 25
  t171 : 25 ≤ 25
  h159_0 : MxxRuntime.sliceMatrix (generatedRoot.w_156_0 ctx) 2 3 0 25 (generatedRoot.w_159_0 ctx)
  t173 : 0 ≤ (3)
  h160_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_160 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_98_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_99_0 ctx), ((generatedRoot.w_147_0 ctx), ((generatedRoot.w_148_0 ctx), ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_157_0 ctx), ((generatedRoot.w_158_0 ctx), ((generatedRoot.w_159_0 ctx), ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_92_0 ctx), ((generatedRoot.w_94_0 ctx), ((generatedRoot.w_96_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (3)) ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_91_0 ctx), ((generatedRoot.w_93_0 ctx), ((generatedRoot.w_95_0 ctx), ((generatedRoot.w_97_0 ctx), ())))))))))) ((generatedRoot.w_160_0 ctx), ((generatedRoot.w_160_1 ctx), ((generatedRoot.w_160_2 ctx), ((generatedRoot.w_160_3 ctx), ((generatedRoot.w_160_4 ctx), ((generatedRoot.w_160_5 ctx), ((generatedRoot.w_160_6 ctx), ((generatedRoot.w_160_7 ctx), ((generatedRoot.w_160_8 ctx), ((generatedRoot.w_160_9 ctx), ()))))))))))
  h161_0 : MxxRuntime.familyGetStatic (generatedRoot.w_98_0 ctx) (3) (generatedRoot.w_161_0 ctx)
  t176 : 0 ≤ (3)
  t177 : (3) < 4
  h163_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_162_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_163_0 ctx)
  h164_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_162_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_164_0 ctx)
  h163_0_1 : MxxRuntime.concatRows (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_concat_1 ctx)
  h165_0 : MxxRuntime.concatRows (generatedRoot.w_165_concat_1 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx)
  h166_0 : MxxRuntime.familyGetStatic (generatedRoot.w_99_0 ctx) (3) (generatedRoot.w_166_0 ctx)
  h167_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_162_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_167_0 ctx)
  h168_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_162_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_168_0 ctx)
  h167_0_1 : MxxRuntime.concatRows (generatedRoot.w_166_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_concat_1 ctx)
  h169_0 : MxxRuntime.concatRows (generatedRoot.w_169_concat_1 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx)
  h170_0 : MxxRuntime.concatColumns (generatedRoot.w_165_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx)
  t188 : 0 ≤ (3)
  t189 : (3) < 4
  h171_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_147_0 ctx) (generatedRoot.w_171_0 ctx)
  t191 : 0 ≤ (3)

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_5 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_160_1 ctx) (generatedRoot.w_160_2 ctx) (generatedRoot.w_160_3 ctx) (generatedRoot.w_160_4 ctx) (generatedRoot.w_160_5 ctx) (generatedRoot.w_160_6 ctx) (generatedRoot.w_160_7 ctx) (generatedRoot.w_160_8 ctx) (generatedRoot.w_160_9 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_165_concat_1 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_169_concat_1 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_171_0 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
  t192 : (3) < 4
  h172_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_148_0 ctx) (generatedRoot.w_172_0 ctx)
  t194 : 0 ≤ (3)
  t195 : (3) < 4
  h173_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_149_0 ctx) (generatedRoot.w_173_0 ctx)
  h172_0_1 : MxxRuntime.concatRows (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_concat_1 ctx)
  h174_0 : MxxRuntime.concatRows (generatedRoot.w_174_concat_1 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx)
  t199 : 0 ≤ (3)
  t200 : (3) < 4
  h175_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_157_0 ctx) (generatedRoot.w_175_0 ctx)
  t202 : 0 ≤ (3)
  t203 : (3) < 4
  h176_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_158_0 ctx) (generatedRoot.w_176_0 ctx)
  t205 : 0 ≤ (3)
  t206 : (3) < 4
  h177_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_159_0 ctx) (generatedRoot.w_177_0 ctx)
  h176_0_1 : MxxRuntime.concatRows (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_concat_1 ctx)
  h178_0 : MxxRuntime.concatRows (generatedRoot.w_178_concat_1 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx)
  h179_0 : MxxRuntime.concatColumns (generatedRoot.w_174_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx)
  h180_0 : MxxRuntime.concatRows (generatedRoot.w_160_6 ctx) (generatedRoot.w_160_8 ctx) (generatedRoot.w_180_0 ctx)
  h181_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx)
  h182_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_181_0 ctx) (generatedRoot.w_182_0 ctx)
  t214 : 0 ≤ 1
  t215 : 1 < 2
  t216 : 2 ≤ 3
  t217 : 0 ≤ 0
  t218 : 0 < 25
  t219 : 25 ≤ 25
  h184_0 : MxxRuntime.sliceMatrix (generatedRoot.w_183_0 ctx) 1 2 0 25 (generatedRoot.w_184_0 ctx)
  t221 : 0 ≤ 2
  t222 : 2 < 3
  t223 : 3 ≤ 3

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_6 backend (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_6 ctx) (generatedRoot.w_160_8 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_174_concat_1 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_178_concat_1 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) : Prop where
  t224 : 0 ≤ 0
  t225 : 0 < 25
  t226 : 25 ≤ 25
  h185_0 : MxxRuntime.sliceMatrix (generatedRoot.w_183_0 ctx) 2 3 0 25 (generatedRoot.w_185_0 ctx)
  h186_0 : MxxRuntime.concatRows (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx)
  h187_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx)
  h188_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx)
  t231 : 0 ≤ 0
  t232 : 0 < 1
  t233 : 1 ≤ 3
  t234 : 0 ≤ 0
  t235 : 0 < 25
  t236 : 25 ≤ 25
  h190_0 : MxxRuntime.sliceMatrix (generatedRoot.w_189_0 ctx) 0 1 0 25 (generatedRoot.w_190_0 ctx)
  t238 : 0 ≤ 0
  t239 : 0 < 1
  t240 : 1 ≤ 3
  t241 : 0 ≤ 0
  t242 : 0 < 25
  t243 : 25 ≤ 25
  h191_0 : MxxRuntime.sliceMatrix (generatedRoot.w_183_0 ctx) 0 1 0 25 (generatedRoot.w_191_0 ctx)
  h194_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_194_0 ctx)
  h197_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_196_0 ctx) (generatedRoot.w_197_0 ctx)
  h198_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx)
  h200_0 : MxxRuntime.concatRows (generatedRoot.w_160_7 ctx) (generatedRoot.w_160_9 ctx) (generatedRoot.w_200_0 ctx)
  h201_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_200_0 ctx) (generatedRoot.w_201_0 ctx)
  h202_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_201_0 ctx) (generatedRoot.w_202_0 ctx)
  t251 : 0 ≤ 1
  t252 : 1 < 2
  t253 : 2 ≤ 3
  t254 : 0 ≤ 0
  t255 : 0 < 25

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_7 backend (generatedRoot.w_74_0 ctx) (generatedRoot.w_160_7 ctx) (generatedRoot.w_160_9 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_202_0 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8  (ctx : generatedRootCtx) : Prop where
  t256 : 25 ≤ 25
  h204_0 : MxxRuntime.sliceMatrix (generatedRoot.w_203_0 ctx) 1 2 0 25 (generatedRoot.w_204_0 ctx)
  t258 : 0 ≤ 2
  t259 : 2 < 3
  t260 : 3 ≤ 3
  t261 : 0 ≤ 0
  t262 : 0 < 25
  t263 : 25 ≤ 25
  h205_0 : MxxRuntime.sliceMatrix (generatedRoot.w_203_0 ctx) 2 3 0 25 (generatedRoot.w_205_0 ctx)
  h206_0 : MxxRuntime.concatRows (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_206_0 ctx)
  h207_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx)
  h208_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx)
  t268 : 0 ≤ 0
  t269 : 0 < 1
  t270 : 1 ≤ 3
  t271 : 0 ≤ 0
  t272 : 0 < 25
  t273 : 25 ≤ 25
  h210_0 : MxxRuntime.sliceMatrix (generatedRoot.w_209_0 ctx) 0 1 0 25 (generatedRoot.w_210_0 ctx)
  t275 : 0 ≤ 0
  t276 : 0 < 1
  t277 : 1 ≤ 3
  t278 : 0 ≤ 0
  t279 : 0 < 25
  t280 : 25 ≤ 25
  h211_0 : MxxRuntime.sliceMatrix (generatedRoot.w_203_0 ctx) 0 1 0 25 (generatedRoot.w_211_0 ctx)
  h215_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_215_0 ctx)
  h218_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_217_0 ctx) (generatedRoot.w_218_0 ctx)
  h219_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_218_0 ctx) (generatedRoot.w_219_0 ctx)
  h221_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_221_0 ctx)
  h224_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx)
  h225_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_224_0 ctx) (generatedRoot.w_225_0 ctx)

theorem generatedRoot.facts_8  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_8 backend (generatedRoot.w_57_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_217_0 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_225_0 ctx)) :
    generatedRoot.Facts_8 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_9  (ctx : generatedRootCtx) : Prop where
  h230_0 : MxxRuntime.familyGetStatic (generatedRoot.w_229_0 ctx) (0) (generatedRoot.w_230_0 ctx)
  h232_0 : MxxRuntime.familyGetStatic (generatedRoot.w_231_0 ctx) (0) (generatedRoot.w_232_0 ctx)
  h234_0 : MxxRuntime.familyGetStatic (generatedRoot.w_233_0 ctx) (0) (generatedRoot.w_234_0 ctx)
  h236_0 : MxxRuntime.familyGetStatic (generatedRoot.w_235_0 ctx) (0) (generatedRoot.w_236_0 ctx)
  t292 : 0 ≤ (0)
  h239_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_239 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_237_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_238_0 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_5 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_231_0 ctx), ((generatedRoot.w_233_0 ctx), ((generatedRoot.w_235_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_230_0 ctx), ((generatedRoot.w_232_0 ctx), ((generatedRoot.w_234_0 ctx), ((generatedRoot.w_236_0 ctx), ())))))))))) ((generatedRoot.w_239_0 ctx), ((generatedRoot.w_239_1 ctx), ((generatedRoot.w_239_2 ctx), ((generatedRoot.w_239_3 ctx), ((generatedRoot.w_239_4 ctx), ((generatedRoot.w_239_5 ctx), ((generatedRoot.w_239_6 ctx), ((generatedRoot.w_239_7 ctx), ((generatedRoot.w_239_8 ctx), ((generatedRoot.w_239_9 ctx), ()))))))))))
  h240_0 : MxxRuntime.familyGetStatic (generatedRoot.w_237_0 ctx) (0) (generatedRoot.w_240_0 ctx)
  t295 : 0 ≤ (0)
  t296 : (0) < 1
  h242_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_241_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_242_0 ctx)
  h243_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_241_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_243_0 ctx)
  h242_0_1 : MxxRuntime.concatRows (generatedRoot.w_240_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_244_concat_1 ctx)
  h244_0 : MxxRuntime.concatRows (generatedRoot.w_244_concat_1 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx)
  h245_0 : MxxRuntime.familyGetStatic (generatedRoot.w_238_0 ctx) (0) (generatedRoot.w_245_0 ctx)
  h246_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_241_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_246_0 ctx)
  h247_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_241_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_247_0 ctx)
  h246_0_1 : MxxRuntime.concatRows (generatedRoot.w_245_0 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_248_concat_1 ctx)
  h248_0 : MxxRuntime.concatRows (generatedRoot.w_248_concat_1 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx)
  h249_0 : MxxRuntime.concatColumns (generatedRoot.w_244_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_249_0 ctx)
  t307 : 0 ≤ (0)
  t308 : (0) < 1
  h250_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_56_4 ctx) (generatedRoot.w_250_0 ctx)
  t310 : 0 ≤ (0)
  t311 : (0) < 1
  h251_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_56_0 ctx) (generatedRoot.w_251_0 ctx)
  t313 : 0 ≤ (0)
  t314 : (0) < 1
  h252_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_56_2 ctx) (generatedRoot.w_252_0 ctx)
  h251_0_1 : MxxRuntime.concatRows (generatedRoot.w_250_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_253_concat_1 ctx)
  h253_0 : MxxRuntime.concatRows (generatedRoot.w_253_concat_1 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_253_0 ctx)
  t318 : 0 ≤ (0)
  t319 : (0) < 1

theorem generatedRoot.facts_9  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_9 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_239_1 ctx) (generatedRoot.w_239_2 ctx) (generatedRoot.w_239_3 ctx) (generatedRoot.w_239_4 ctx) (generatedRoot.w_239_5 ctx) (generatedRoot.w_239_6 ctx) (generatedRoot.w_239_7 ctx) (generatedRoot.w_239_8 ctx) (generatedRoot.w_239_9 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_241_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_244_concat_1 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_248_concat_1 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_253_concat_1 ctx)) :
    generatedRoot.Facts_9 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_10  (ctx : generatedRootCtx) : Prop where
  h254_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_56_5 ctx) (generatedRoot.w_254_0 ctx)
  t321 : 0 ≤ (0)
  t322 : (0) < 1
  h255_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_56_1 ctx) (generatedRoot.w_255_0 ctx)
  t324 : 0 ≤ (0)
  t325 : (0) < 1
  h256_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_56_3 ctx) (generatedRoot.w_256_0 ctx)
  h255_0_1 : MxxRuntime.concatRows (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_257_concat_1 ctx)
  h257_0 : MxxRuntime.concatRows (generatedRoot.w_257_concat_1 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx)
  h258_0 : MxxRuntime.concatColumns (generatedRoot.w_253_0 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_258_0 ctx)
  h259_0 : MxxRuntime.concatRows (generatedRoot.w_239_7 ctx) (generatedRoot.w_239_9 ctx) (generatedRoot.w_259_0 ctx)
  h260_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx)
  h261_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx)
  t333 : 0 ≤ 1
  t334 : 1 < 2
  t335 : 2 ≤ 3
  t336 : 0 ≤ 0
  t337 : 0 < 25
  t338 : 25 ≤ 25
  h263_0 : MxxRuntime.sliceMatrix (generatedRoot.w_262_0 ctx) 1 2 0 25 (generatedRoot.w_263_0 ctx)
  t340 : 0 ≤ 2
  t341 : 2 < 3
  t342 : 3 ≤ 3
  t343 : 0 ≤ 0
  t344 : 0 < 25
  t345 : 25 ≤ 25
  h264_0 : MxxRuntime.sliceMatrix (generatedRoot.w_262_0 ctx) 2 3 0 25 (generatedRoot.w_264_0 ctx)
  h265_0 : MxxRuntime.concatRows (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx)
  h266_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_265_0 ctx) (generatedRoot.w_266_0 ctx)
  h267_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_266_0 ctx) (generatedRoot.w_267_0 ctx)
  t350 : 0 ≤ 0
  t351 : 0 < 1

theorem generatedRoot.facts_10  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_10 backend (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_239_7 ctx) (generatedRoot.w_239_9 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_257_concat_1 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_266_0 ctx) (generatedRoot.w_267_0 ctx)) :
    generatedRoot.Facts_10 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_11  (ctx : generatedRootCtx) : Prop where
  t352 : 1 ≤ 3
  t353 : 0 ≤ 0
  t354 : 0 < 25
  t355 : 25 ≤ 25
  h269_0 : MxxRuntime.sliceMatrix (generatedRoot.w_268_0 ctx) 0 1 0 25 (generatedRoot.w_269_0 ctx)
  t357 : 0 ≤ 0
  t358 : 0 < 1
  t359 : 1 ≤ 3
  t360 : 0 ≤ 0
  t361 : 0 < 25
  t362 : 25 ≤ 25
  h270_0 : MxxRuntime.sliceMatrix (generatedRoot.w_262_0 ctx) 0 1 0 25 (generatedRoot.w_270_0 ctx)
  h273_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_273_0 ctx)
  h276_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx)
  h277_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx)
  h279_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_279_0 ctx)
  h282_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_281_0 ctx) (generatedRoot.w_282_0 ctx)
  h283_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_282_0 ctx) (generatedRoot.w_283_0 ctx)
  h288_0 : MxxRuntime.familyGetStatic (generatedRoot.w_287_0 ctx) (0) (generatedRoot.w_288_0 ctx)
  h290_0 : MxxRuntime.familyGetStatic (generatedRoot.w_289_0 ctx) (0) (generatedRoot.w_290_0 ctx)
  h292_0 : MxxRuntime.familyGetStatic (generatedRoot.w_291_0 ctx) (0) (generatedRoot.w_292_0 ctx)
  h294_0 : MxxRuntime.familyGetStatic (generatedRoot.w_293_0 ctx) (0) (generatedRoot.w_294_0 ctx)
  t374 : 0 ≤ (0)
  h297_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_297 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_295_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_296_0 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_5 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_289_0 ctx), ((generatedRoot.w_291_0 ctx), ((generatedRoot.w_293_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_286_0 ctx), ((generatedRoot.w_286_0 ctx), ((generatedRoot.w_286_0 ctx), ((generatedRoot.w_286_0 ctx), ((generatedRoot.w_286_0 ctx), ((generatedRoot.w_286_0 ctx), ((generatedRoot.w_288_0 ctx), ((generatedRoot.w_290_0 ctx), ((generatedRoot.w_292_0 ctx), ((generatedRoot.w_294_0 ctx), ())))))))))) ((generatedRoot.w_297_0 ctx), ((generatedRoot.w_297_1 ctx), ((generatedRoot.w_297_2 ctx), ((generatedRoot.w_297_3 ctx), ((generatedRoot.w_297_4 ctx), ((generatedRoot.w_297_5 ctx), ((generatedRoot.w_297_6 ctx), ((generatedRoot.w_297_7 ctx), ((generatedRoot.w_297_8 ctx), ((generatedRoot.w_297_9 ctx), ()))))))))))
  h298_0 : MxxRuntime.familyGetStatic (generatedRoot.w_295_0 ctx) (0) (generatedRoot.w_298_0 ctx)
  t377 : 0 ≤ (0)
  t378 : (0) < 1
  h300_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_299_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_300_0 ctx)
  h301_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_299_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_301_0 ctx)
  h300_0_1 : MxxRuntime.concatRows (generatedRoot.w_298_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_302_concat_1 ctx)
  h302_0 : MxxRuntime.concatRows (generatedRoot.w_302_concat_1 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_302_0 ctx)
  h303_0 : MxxRuntime.familyGetStatic (generatedRoot.w_296_0 ctx) (0) (generatedRoot.w_303_0 ctx)

theorem generatedRoot.facts_11  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_11 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_289_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_270_0 ctx) (generatedRoot.w_273_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_281_0 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_286_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_297_1 ctx) (generatedRoot.w_297_2 ctx) (generatedRoot.w_297_3 ctx) (generatedRoot.w_297_4 ctx) (generatedRoot.w_297_5 ctx) (generatedRoot.w_297_6 ctx) (generatedRoot.w_297_7 ctx) (generatedRoot.w_297_8 ctx) (generatedRoot.w_297_9 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_302_concat_1 ctx) (generatedRoot.w_303_0 ctx)) :
    generatedRoot.Facts_11 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_12  (ctx : generatedRootCtx) : Prop where
  h304_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_299_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_304_0 ctx)
  h305_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_299_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_305_0 ctx)
  h304_0_1 : MxxRuntime.concatRows (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_306_concat_1 ctx)
  h306_0 : MxxRuntime.concatRows (generatedRoot.w_306_concat_1 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_306_0 ctx)
  h307_0 : MxxRuntime.concatColumns (generatedRoot.w_302_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_307_0 ctx)
  t389 : 0 ≤ (0)
  t390 : (0) < 1
  h308_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_56_4 ctx) (generatedRoot.w_308_0 ctx)
  t392 : 0 ≤ (0)
  t393 : (0) < 1
  h309_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_56_0 ctx) (generatedRoot.w_309_0 ctx)
  t395 : 0 ≤ (0)
  t396 : (0) < 1
  h310_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_56_2 ctx) (generatedRoot.w_310_0 ctx)
  h309_0_1 : MxxRuntime.concatRows (generatedRoot.w_308_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_311_concat_1 ctx)
  h311_0 : MxxRuntime.concatRows (generatedRoot.w_311_concat_1 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_311_0 ctx)
  t400 : 0 ≤ (0)
  t401 : (0) < 1
  h312_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_56_5 ctx) (generatedRoot.w_312_0 ctx)
  t403 : 0 ≤ (0)
  t404 : (0) < 1
  h313_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_56_1 ctx) (generatedRoot.w_313_0 ctx)
  t406 : 0 ≤ (0)
  t407 : (0) < 1
  h314_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_56_3 ctx) (generatedRoot.w_314_0 ctx)
  h313_0_1 : MxxRuntime.concatRows (generatedRoot.w_312_0 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_315_concat_1 ctx)
  h315_0 : MxxRuntime.concatRows (generatedRoot.w_315_concat_1 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_315_0 ctx)
  h316_0 : MxxRuntime.concatColumns (generatedRoot.w_311_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx)
  h317_0 : MxxRuntime.concatRows (generatedRoot.w_297_7 ctx) (generatedRoot.w_297_9 ctx) (generatedRoot.w_317_0 ctx)
  h318_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_317_0 ctx) (generatedRoot.w_318_0 ctx)
  h319_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx)
  t415 : 0 ≤ 1

theorem generatedRoot.facts_12  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_12 backend hashModel (generatedRoot.w_7_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_297_7 ctx) (generatedRoot.w_297_9 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_306_concat_1 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_311_concat_1 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_315_concat_1 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_317_0 ctx) (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx)) :
    generatedRoot.Facts_12 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_13  (ctx : generatedRootCtx) : Prop where
  t416 : 1 < 2
  t417 : 2 ≤ 3
  t418 : 0 ≤ 0
  t419 : 0 < 25
  t420 : 25 ≤ 25
  h321_0 : MxxRuntime.sliceMatrix (generatedRoot.w_320_0 ctx) 1 2 0 25 (generatedRoot.w_321_0 ctx)
  t422 : 0 ≤ 2
  t423 : 2 < 3
  t424 : 3 ≤ 3
  t425 : 0 ≤ 0
  t426 : 0 < 25
  t427 : 25 ≤ 25
  h322_0 : MxxRuntime.sliceMatrix (generatedRoot.w_320_0 ctx) 2 3 0 25 (generatedRoot.w_322_0 ctx)
  h323_0 : MxxRuntime.concatRows (generatedRoot.w_321_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_323_0 ctx)
  h324_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx)
  h325_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_324_0 ctx) (generatedRoot.w_325_0 ctx)
  t432 : 0 ≤ 0
  t433 : 0 < 1
  t434 : 1 ≤ 3
  t435 : 0 ≤ 0
  t436 : 0 < 25
  t437 : 25 ≤ 25
  h327_0 : MxxRuntime.sliceMatrix (generatedRoot.w_326_0 ctx) 0 1 0 25 (generatedRoot.w_327_0 ctx)
  t439 : 0 ≤ 0
  t440 : 0 < 1
  t441 : 1 ≤ 3
  t442 : 0 ≤ 0
  t443 : 0 < 25
  t444 : 25 ≤ 25
  h328_0 : MxxRuntime.sliceMatrix (generatedRoot.w_320_0 ctx) 0 1 0 25 (generatedRoot.w_328_0 ctx)
  h331_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_331_0 ctx)
  h334_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_333_0 ctx) (generatedRoot.w_334_0 ctx)

theorem generatedRoot.facts_13  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_13 backend (generatedRoot.w_57_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_325_0 ctx) (generatedRoot.w_326_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_333_0 ctx) (generatedRoot.w_334_0 ctx)) :
    generatedRoot.Facts_13 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_14  (ctx : generatedRootCtx) : Prop where
  h335_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx)
  h337_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_337_0 ctx)
  h340_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx)
  h341_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_340_0 ctx) (generatedRoot.w_341_0 ctx)
  h346_0 : MxxRuntime.familyGetStatic (generatedRoot.w_345_0 ctx) (0) (generatedRoot.w_346_0 ctx)
  h348_0 : MxxRuntime.familyGetStatic (generatedRoot.w_347_0 ctx) (0) (generatedRoot.w_348_0 ctx)
  h350_0 : MxxRuntime.familyGetStatic (generatedRoot.w_349_0 ctx) (0) (generatedRoot.w_350_0 ctx)
  h352_0 : MxxRuntime.familyGetStatic (generatedRoot.w_351_0 ctx) (0) (generatedRoot.w_352_0 ctx)
  t456 : 0 ≤ (0)
  h355_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_355 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_353_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_354_0 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_5 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_347_0 ctx), ((generatedRoot.w_349_0 ctx), ((generatedRoot.w_351_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_346_0 ctx), ((generatedRoot.w_348_0 ctx), ((generatedRoot.w_350_0 ctx), ((generatedRoot.w_352_0 ctx), ())))))))))) ((generatedRoot.w_355_0 ctx), ((generatedRoot.w_355_1 ctx), ((generatedRoot.w_355_2 ctx), ((generatedRoot.w_355_3 ctx), ((generatedRoot.w_355_4 ctx), ((generatedRoot.w_355_5 ctx), ((generatedRoot.w_355_6 ctx), ((generatedRoot.w_355_7 ctx), ((generatedRoot.w_355_8 ctx), ((generatedRoot.w_355_9 ctx), ()))))))))))
  h356_0 : MxxRuntime.familyGetStatic (generatedRoot.w_353_0 ctx) (0) (generatedRoot.w_356_0 ctx)
  t459 : 0 ≤ (0)
  t460 : (0) < 1
  h358_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_357_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_358_0 ctx)
  h359_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_357_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_359_0 ctx)
  h358_0_1 : MxxRuntime.concatRows (generatedRoot.w_356_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_360_concat_1 ctx)
  h360_0 : MxxRuntime.concatRows (generatedRoot.w_360_concat_1 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx)
  h361_0 : MxxRuntime.familyGetStatic (generatedRoot.w_354_0 ctx) (0) (generatedRoot.w_361_0 ctx)
  h362_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_357_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_362_0 ctx)
  h363_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_357_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_363_0 ctx)
  h362_0_1 : MxxRuntime.concatRows (generatedRoot.w_361_0 ctx) (generatedRoot.w_362_0 ctx) (generatedRoot.w_364_concat_1 ctx)
  h364_0 : MxxRuntime.concatRows (generatedRoot.w_364_concat_1 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx)
  h365_0 : MxxRuntime.concatColumns (generatedRoot.w_360_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_365_0 ctx)
  t471 : 0 ≤ (0)
  t472 : (0) < 1
  h366_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_56_4 ctx) (generatedRoot.w_366_0 ctx)
  t474 : 0 ≤ (0)
  t475 : (0) < 1
  h367_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_56_0 ctx) (generatedRoot.w_367_0 ctx)
  t477 : 0 ≤ (0)
  t478 : (0) < 1
  h368_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_56_2 ctx) (generatedRoot.w_368_0 ctx)

theorem generatedRoot.facts_14  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_14 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_349_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_353_0 ctx) (generatedRoot.w_354_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_337_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_341_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_346_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_350_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_355_1 ctx) (generatedRoot.w_355_2 ctx) (generatedRoot.w_355_3 ctx) (generatedRoot.w_355_4 ctx) (generatedRoot.w_355_5 ctx) (generatedRoot.w_355_6 ctx) (generatedRoot.w_355_7 ctx) (generatedRoot.w_355_8 ctx) (generatedRoot.w_355_9 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_357_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_360_concat_1 ctx) (generatedRoot.w_361_0 ctx) (generatedRoot.w_362_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_364_concat_1 ctx) (generatedRoot.w_365_0 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx)) :
    generatedRoot.Facts_14 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_15  (ctx : generatedRootCtx) : Prop where
  h367_0_1 : MxxRuntime.concatRows (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_369_concat_1 ctx)
  h369_0 : MxxRuntime.concatRows (generatedRoot.w_369_concat_1 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_369_0 ctx)
  t482 : 0 ≤ (0)
  t483 : (0) < 1
  h370_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_56_5 ctx) (generatedRoot.w_370_0 ctx)
  t485 : 0 ≤ (0)
  t486 : (0) < 1
  h371_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_56_1 ctx) (generatedRoot.w_371_0 ctx)
  t488 : 0 ≤ (0)
  t489 : (0) < 1
  h372_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_56_3 ctx) (generatedRoot.w_372_0 ctx)
  h371_0_1 : MxxRuntime.concatRows (generatedRoot.w_370_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_373_concat_1 ctx)
  h373_0 : MxxRuntime.concatRows (generatedRoot.w_373_concat_1 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_373_0 ctx)
  h374_0 : MxxRuntime.concatColumns (generatedRoot.w_369_0 ctx) (generatedRoot.w_373_0 ctx) (generatedRoot.w_374_0 ctx)
  h375_0 : MxxRuntime.concatRows (generatedRoot.w_355_7 ctx) (generatedRoot.w_355_9 ctx) (generatedRoot.w_375_0 ctx)
  h376_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx)
  h377_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_376_0 ctx) (generatedRoot.w_377_0 ctx)
  t497 : 0 ≤ 1
  t498 : 1 < 2
  t499 : 2 ≤ 3
  t500 : 0 ≤ 0
  t501 : 0 < 25
  t502 : 25 ≤ 25
  h379_0 : MxxRuntime.sliceMatrix (generatedRoot.w_378_0 ctx) 1 2 0 25 (generatedRoot.w_379_0 ctx)
  t504 : 0 ≤ 2
  t505 : 2 < 3
  t506 : 3 ≤ 3
  t507 : 0 ≤ 0
  t508 : 0 < 25
  t509 : 25 ≤ 25
  h380_0 : MxxRuntime.sliceMatrix (generatedRoot.w_378_0 ctx) 2 3 0 25 (generatedRoot.w_380_0 ctx)
  h381_0 : MxxRuntime.concatRows (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_381_0 ctx)

theorem generatedRoot.facts_15  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_15 backend (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_355_7 ctx) (generatedRoot.w_355_9 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_369_0 ctx) (generatedRoot.w_369_concat_1 ctx) (generatedRoot.w_370_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_373_0 ctx) (generatedRoot.w_373_concat_1 ctx) (generatedRoot.w_374_0 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_377_0 ctx) (generatedRoot.w_378_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_381_0 ctx)) :
    generatedRoot.Facts_15 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_16  (ctx : generatedRootCtx) : Prop where
  h382_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_381_0 ctx) (generatedRoot.w_382_0 ctx)
  h383_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_382_0 ctx) (generatedRoot.w_383_0 ctx)
  t514 : 0 ≤ 0
  t515 : 0 < 1
  t516 : 1 ≤ 3
  t517 : 0 ≤ 0
  t518 : 0 < 25
  t519 : 25 ≤ 25
  h385_0 : MxxRuntime.sliceMatrix (generatedRoot.w_384_0 ctx) 0 1 0 25 (generatedRoot.w_385_0 ctx)
  t521 : 0 ≤ 0
  t522 : 0 < 1
  t523 : 1 ≤ 3
  t524 : 0 ≤ 0
  t525 : 0 < 25
  t526 : 25 ≤ 25
  h386_0 : MxxRuntime.sliceMatrix (generatedRoot.w_378_0 ctx) 0 1 0 25 (generatedRoot.w_386_0 ctx)
  h389_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_389_0 ctx)
  h392_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx)
  h393_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_392_0 ctx) (generatedRoot.w_393_0 ctx)
  h395_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_395_0 ctx)
  h398_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_397_0 ctx) (generatedRoot.w_398_0 ctx)
  h399_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_398_0 ctx) (generatedRoot.w_399_0 ctx)
  h404_0 : MxxRuntime.familyGetStatic (generatedRoot.w_403_0 ctx) (0) (generatedRoot.w_404_0 ctx)
  h406_0 : MxxRuntime.familyGetStatic (generatedRoot.w_405_0 ctx) (0) (generatedRoot.w_406_0 ctx)
  h408_0 : MxxRuntime.familyGetStatic (generatedRoot.w_407_0 ctx) (0) (generatedRoot.w_408_0 ctx)
  h410_0 : MxxRuntime.familyGetStatic (generatedRoot.w_409_0 ctx) (0) (generatedRoot.w_410_0 ctx)
  t538 : 0 ≤ (0)
  h413_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_413 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_411_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_412_0 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_5 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_403_0 ctx), ((generatedRoot.w_405_0 ctx), ((generatedRoot.w_407_0 ctx), ((generatedRoot.w_409_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_402_0 ctx), ((generatedRoot.w_402_0 ctx), ((generatedRoot.w_402_0 ctx), ((generatedRoot.w_402_0 ctx), ((generatedRoot.w_402_0 ctx), ((generatedRoot.w_402_0 ctx), ((generatedRoot.w_404_0 ctx), ((generatedRoot.w_406_0 ctx), ((generatedRoot.w_408_0 ctx), ((generatedRoot.w_410_0 ctx), ())))))))))) ((generatedRoot.w_413_0 ctx), ((generatedRoot.w_413_1 ctx), ((generatedRoot.w_413_2 ctx), ((generatedRoot.w_413_3 ctx), ((generatedRoot.w_413_4 ctx), ((generatedRoot.w_413_5 ctx), ((generatedRoot.w_413_6 ctx), ((generatedRoot.w_413_7 ctx), ((generatedRoot.w_413_8 ctx), ((generatedRoot.w_413_9 ctx), ()))))))))))
  h414_0 : MxxRuntime.familyGetStatic (generatedRoot.w_411_0 ctx) (0) (generatedRoot.w_414_0 ctx)
  t541 : 0 ≤ (0)
  t542 : (0) < 1
  h416_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_415_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_416_0 ctx)

theorem generatedRoot.facts_16  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_16 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_405_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_409_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_378_0 ctx) (generatedRoot.w_381_0 ctx) (generatedRoot.w_382_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_385_0 ctx) (generatedRoot.w_386_0 ctx) (generatedRoot.w_389_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_393_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_397_0 ctx) (generatedRoot.w_398_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_402_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_406_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_410_0 ctx) (generatedRoot.w_413_0 ctx) (generatedRoot.w_413_1 ctx) (generatedRoot.w_413_2 ctx) (generatedRoot.w_413_3 ctx) (generatedRoot.w_413_4 ctx) (generatedRoot.w_413_5 ctx) (generatedRoot.w_413_6 ctx) (generatedRoot.w_413_7 ctx) (generatedRoot.w_413_8 ctx) (generatedRoot.w_413_9 ctx) (generatedRoot.w_414_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx)) :
    generatedRoot.Facts_16 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_17  (ctx : generatedRootCtx) : Prop where
  h417_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_415_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_417_0 ctx)
  h416_0_1 : MxxRuntime.concatRows (generatedRoot.w_414_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_418_concat_1 ctx)
  h418_0 : MxxRuntime.concatRows (generatedRoot.w_418_concat_1 ctx) (generatedRoot.w_417_0 ctx) (generatedRoot.w_418_0 ctx)
  h419_0 : MxxRuntime.familyGetStatic (generatedRoot.w_412_0 ctx) (0) (generatedRoot.w_419_0 ctx)
  h420_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_415_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_420_0 ctx)
  h421_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_415_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_421_0 ctx)
  h420_0_1 : MxxRuntime.concatRows (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_422_concat_1 ctx)
  h422_0 : MxxRuntime.concatRows (generatedRoot.w_422_concat_1 ctx) (generatedRoot.w_421_0 ctx) (generatedRoot.w_422_0 ctx)
  h423_0 : MxxRuntime.concatColumns (generatedRoot.w_418_0 ctx) (generatedRoot.w_422_0 ctx) (generatedRoot.w_423_0 ctx)
  t553 : 0 ≤ (0)
  t554 : (0) < 1
  h424_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_56_4 ctx) (generatedRoot.w_424_0 ctx)
  t556 : 0 ≤ (0)
  t557 : (0) < 1
  h425_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_56_0 ctx) (generatedRoot.w_425_0 ctx)
  t559 : 0 ≤ (0)
  t560 : (0) < 1
  h426_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_56_2 ctx) (generatedRoot.w_426_0 ctx)
  h425_0_1 : MxxRuntime.concatRows (generatedRoot.w_424_0 ctx) (generatedRoot.w_425_0 ctx) (generatedRoot.w_427_concat_1 ctx)
  h427_0 : MxxRuntime.concatRows (generatedRoot.w_427_concat_1 ctx) (generatedRoot.w_426_0 ctx) (generatedRoot.w_427_0 ctx)
  t564 : 0 ≤ (0)
  t565 : (0) < 1
  h428_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_56_5 ctx) (generatedRoot.w_428_0 ctx)
  t567 : 0 ≤ (0)
  t568 : (0) < 1
  h429_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_56_1 ctx) (generatedRoot.w_429_0 ctx)
  t570 : 0 ≤ (0)
  t571 : (0) < 1
  h430_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_56_3 ctx) (generatedRoot.w_430_0 ctx)
  h429_0_1 : MxxRuntime.concatRows (generatedRoot.w_428_0 ctx) (generatedRoot.w_429_0 ctx) (generatedRoot.w_431_concat_1 ctx)
  h431_0 : MxxRuntime.concatRows (generatedRoot.w_431_concat_1 ctx) (generatedRoot.w_430_0 ctx) (generatedRoot.w_431_0 ctx)
  h432_0 : MxxRuntime.concatColumns (generatedRoot.w_427_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx)

theorem generatedRoot.facts_17  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_17 hashModel (generatedRoot.w_7_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_414_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_417_0 ctx) (generatedRoot.w_418_0 ctx) (generatedRoot.w_418_concat_1 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_421_0 ctx) (generatedRoot.w_422_0 ctx) (generatedRoot.w_422_concat_1 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_425_0 ctx) (generatedRoot.w_426_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_427_concat_1 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_429_0 ctx) (generatedRoot.w_430_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_431_concat_1 ctx) (generatedRoot.w_432_0 ctx)) :
    generatedRoot.Facts_17 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_18  (ctx : generatedRootCtx) : Prop where
  h433_0 : MxxRuntime.concatRows (generatedRoot.w_413_7 ctx) (generatedRoot.w_413_9 ctx) (generatedRoot.w_433_0 ctx)
  h434_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_433_0 ctx) (generatedRoot.w_434_0 ctx)
  h435_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_434_0 ctx) (generatedRoot.w_435_0 ctx)
  t579 : 0 ≤ 1
  t580 : 1 < 2
  t581 : 2 ≤ 3
  t582 : 0 ≤ 0
  t583 : 0 < 25
  t584 : 25 ≤ 25
  h437_0 : MxxRuntime.sliceMatrix (generatedRoot.w_436_0 ctx) 1 2 0 25 (generatedRoot.w_437_0 ctx)
  t586 : 0 ≤ 2
  t587 : 2 < 3
  t588 : 3 ≤ 3
  t589 : 0 ≤ 0
  t590 : 0 < 25
  t591 : 25 ≤ 25
  h438_0 : MxxRuntime.sliceMatrix (generatedRoot.w_436_0 ctx) 2 3 0 25 (generatedRoot.w_438_0 ctx)
  h439_0 : MxxRuntime.concatRows (generatedRoot.w_437_0 ctx) (generatedRoot.w_438_0 ctx) (generatedRoot.w_439_0 ctx)
  h440_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx)
  h441_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_440_0 ctx) (generatedRoot.w_441_0 ctx)
  t596 : 0 ≤ 0
  t597 : 0 < 1
  t598 : 1 ≤ 3
  t599 : 0 ≤ 0
  t600 : 0 < 25
  t601 : 25 ≤ 25
  h443_0 : MxxRuntime.sliceMatrix (generatedRoot.w_442_0 ctx) 0 1 0 25 (generatedRoot.w_443_0 ctx)
  t603 : 0 ≤ 0
  t604 : 0 < 1
  t605 : 1 ≤ 3
  t606 : 0 ≤ 0
  t607 : 0 < 25

theorem generatedRoot.facts_18  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_18 backend (generatedRoot.w_413_7 ctx) (generatedRoot.w_413_9 ctx) (generatedRoot.w_433_0 ctx) (generatedRoot.w_434_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_437_0 ctx) (generatedRoot.w_438_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_441_0 ctx) (generatedRoot.w_442_0 ctx) (generatedRoot.w_443_0 ctx)) :
    generatedRoot.Facts_18 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_19  (ctx : generatedRootCtx) : Prop where
  t608 : 25 ≤ 25
  h444_0 : MxxRuntime.sliceMatrix (generatedRoot.w_436_0 ctx) 0 1 0 25 (generatedRoot.w_444_0 ctx)
  h447_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_447_0 ctx)
  h450_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_449_0 ctx) (generatedRoot.w_450_0 ctx)
  h451_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_450_0 ctx) (generatedRoot.w_451_0 ctx)
  h453_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_453_0 ctx)
  h456_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx)
  h457_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_456_0 ctx) (generatedRoot.w_457_0 ctx)
  h462_0 : MxxRuntime.familyGetStatic (generatedRoot.w_461_0 ctx) (0) (generatedRoot.w_462_0 ctx)
  h464_0 : MxxRuntime.familyGetStatic (generatedRoot.w_463_0 ctx) (0) (generatedRoot.w_464_0 ctx)
  h466_0 : MxxRuntime.familyGetStatic (generatedRoot.w_465_0 ctx) (0) (generatedRoot.w_466_0 ctx)
  h468_0 : MxxRuntime.familyGetStatic (generatedRoot.w_467_0 ctx) (0) (generatedRoot.w_468_0 ctx)
  t620 : 0 ≤ (0)
  h471_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_471 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_469_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_470_0 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_5 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_461_0 ctx), ((generatedRoot.w_463_0 ctx), ((generatedRoot.w_465_0 ctx), ((generatedRoot.w_467_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_460_0 ctx), ((generatedRoot.w_460_0 ctx), ((generatedRoot.w_460_0 ctx), ((generatedRoot.w_460_0 ctx), ((generatedRoot.w_460_0 ctx), ((generatedRoot.w_460_0 ctx), ((generatedRoot.w_462_0 ctx), ((generatedRoot.w_464_0 ctx), ((generatedRoot.w_466_0 ctx), ((generatedRoot.w_468_0 ctx), ())))))))))) ((generatedRoot.w_471_0 ctx), ((generatedRoot.w_471_1 ctx), ((generatedRoot.w_471_2 ctx), ((generatedRoot.w_471_3 ctx), ((generatedRoot.w_471_4 ctx), ((generatedRoot.w_471_5 ctx), ((generatedRoot.w_471_6 ctx), ((generatedRoot.w_471_7 ctx), ((generatedRoot.w_471_8 ctx), ((generatedRoot.w_471_9 ctx), ()))))))))))
  h472_0 : MxxRuntime.familyGetStatic (generatedRoot.w_469_0 ctx) (0) (generatedRoot.w_472_0 ctx)
  t623 : 0 ≤ (0)
  t624 : (0) < 1
  h474_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_473_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_474_0 ctx)
  h475_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_473_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_475_0 ctx)
  h474_0_1 : MxxRuntime.concatRows (generatedRoot.w_472_0 ctx) (generatedRoot.w_474_0 ctx) (generatedRoot.w_476_concat_1 ctx)
  h476_0 : MxxRuntime.concatRows (generatedRoot.w_476_concat_1 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx)
  h477_0 : MxxRuntime.familyGetStatic (generatedRoot.w_470_0 ctx) (0) (generatedRoot.w_477_0 ctx)
  h478_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_473_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_478_0 ctx)
  h479_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_473_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_479_0 ctx)
  h478_0_1 : MxxRuntime.concatRows (generatedRoot.w_477_0 ctx) (generatedRoot.w_478_0 ctx) (generatedRoot.w_480_concat_1 ctx)
  h480_0 : MxxRuntime.concatRows (generatedRoot.w_480_concat_1 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx)
  h481_0 : MxxRuntime.concatColumns (generatedRoot.w_476_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_481_0 ctx)
  t635 : 0 ≤ (0)
  t636 : (0) < 1
  h482_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_56_4 ctx) (generatedRoot.w_482_0 ctx)
  t638 : 0 ≤ (0)
  t639 : (0) < 1

theorem generatedRoot.facts_19  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_19 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_461_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_465_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_469_0 ctx) (generatedRoot.w_470_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_449_0 ctx) (generatedRoot.w_450_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_453_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_457_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_462_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_466_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_471_1 ctx) (generatedRoot.w_471_2 ctx) (generatedRoot.w_471_3 ctx) (generatedRoot.w_471_4 ctx) (generatedRoot.w_471_5 ctx) (generatedRoot.w_471_6 ctx) (generatedRoot.w_471_7 ctx) (generatedRoot.w_471_8 ctx) (generatedRoot.w_471_9 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_473_0 ctx) (generatedRoot.w_474_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_476_concat_1 ctx) (generatedRoot.w_477_0 ctx) (generatedRoot.w_478_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_480_concat_1 ctx) (generatedRoot.w_481_0 ctx) (generatedRoot.w_482_0 ctx)) :
    generatedRoot.Facts_19 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_20  (ctx : generatedRootCtx) : Prop where
  h483_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_56_0 ctx) (generatedRoot.w_483_0 ctx)
  t641 : 0 ≤ (0)
  t642 : (0) < 1
  h484_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_56_2 ctx) (generatedRoot.w_484_0 ctx)
  h483_0_1 : MxxRuntime.concatRows (generatedRoot.w_482_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_485_concat_1 ctx)
  h485_0 : MxxRuntime.concatRows (generatedRoot.w_485_concat_1 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_485_0 ctx)
  t646 : 0 ≤ (0)
  t647 : (0) < 1
  h486_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_56_5 ctx) (generatedRoot.w_486_0 ctx)
  t649 : 0 ≤ (0)
  t650 : (0) < 1
  h487_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_56_1 ctx) (generatedRoot.w_487_0 ctx)
  t652 : 0 ≤ (0)
  t653 : (0) < 1
  h488_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_56_3 ctx) (generatedRoot.w_488_0 ctx)
  h487_0_1 : MxxRuntime.concatRows (generatedRoot.w_486_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_489_concat_1 ctx)
  h489_0 : MxxRuntime.concatRows (generatedRoot.w_489_concat_1 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_489_0 ctx)
  h490_0 : MxxRuntime.concatColumns (generatedRoot.w_485_0 ctx) (generatedRoot.w_489_0 ctx) (generatedRoot.w_490_0 ctx)
  h491_0 : MxxRuntime.concatRows (generatedRoot.w_471_7 ctx) (generatedRoot.w_471_9 ctx) (generatedRoot.w_491_0 ctx)
  h492_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx)
  h493_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_492_0 ctx) (generatedRoot.w_493_0 ctx)
  t661 : 0 ≤ 1
  t662 : 1 < 2
  t663 : 2 ≤ 3
  t664 : 0 ≤ 0
  t665 : 0 < 25
  t666 : 25 ≤ 25
  h495_0 : MxxRuntime.sliceMatrix (generatedRoot.w_494_0 ctx) 1 2 0 25 (generatedRoot.w_495_0 ctx)
  t668 : 0 ≤ 2
  t669 : 2 < 3
  t670 : 3 ≤ 3
  t671 : 0 ≤ 0

theorem generatedRoot.facts_20  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_20 backend (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_471_7 ctx) (generatedRoot.w_471_9 ctx) (generatedRoot.w_482_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_485_0 ctx) (generatedRoot.w_485_concat_1 ctx) (generatedRoot.w_486_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_489_0 ctx) (generatedRoot.w_489_concat_1 ctx) (generatedRoot.w_490_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_493_0 ctx) (generatedRoot.w_494_0 ctx) (generatedRoot.w_495_0 ctx)) :
    generatedRoot.Facts_20 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_21  (ctx : generatedRootCtx) : Prop where
  t672 : 0 < 25
  t673 : 25 ≤ 25
  h496_0 : MxxRuntime.sliceMatrix (generatedRoot.w_494_0 ctx) 2 3 0 25 (generatedRoot.w_496_0 ctx)
  h497_0 : MxxRuntime.concatRows (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_497_0 ctx)
  h498_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_497_0 ctx) (generatedRoot.w_498_0 ctx)
  h499_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_498_0 ctx) (generatedRoot.w_499_0 ctx)
  t678 : 0 ≤ 0
  t679 : 0 < 1
  t680 : 1 ≤ 3
  t681 : 0 ≤ 0
  t682 : 0 < 25
  t683 : 25 ≤ 25
  h501_0 : MxxRuntime.sliceMatrix (generatedRoot.w_500_0 ctx) 0 1 0 25 (generatedRoot.w_501_0 ctx)
  t685 : 0 ≤ 0
  t686 : 0 < 1
  t687 : 1 ≤ 3
  t688 : 0 ≤ 0
  t689 : 0 < 25
  t690 : 25 ≤ 25
  h502_0 : MxxRuntime.sliceMatrix (generatedRoot.w_494_0 ctx) 0 1 0 25 (generatedRoot.w_502_0 ctx)
  h505_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_505_0 ctx)
  h508_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx)
  h509_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_508_0 ctx) (generatedRoot.w_509_0 ctx)
  h511_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_511_0 ctx)
  h514_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_513_0 ctx) (generatedRoot.w_514_0 ctx)
  h515_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_514_0 ctx) (generatedRoot.w_515_0 ctx)
  h520_0 : MxxRuntime.familyGetStatic (generatedRoot.w_519_0 ctx) (0) (generatedRoot.w_520_0 ctx)
  h522_0 : MxxRuntime.familyGetStatic (generatedRoot.w_521_0 ctx) (0) (generatedRoot.w_522_0 ctx)
  h524_0 : MxxRuntime.familyGetStatic (generatedRoot.w_523_0 ctx) (0) (generatedRoot.w_524_0 ctx)
  h526_0 : MxxRuntime.familyGetStatic (generatedRoot.w_525_0 ctx) (0) (generatedRoot.w_526_0 ctx)
  t702 : 0 ≤ (0)
  h529_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_529 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_527_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_528_0 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_5 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_519_0 ctx), ((generatedRoot.w_521_0 ctx), ((generatedRoot.w_523_0 ctx), ((generatedRoot.w_525_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_518_0 ctx), ((generatedRoot.w_518_0 ctx), ((generatedRoot.w_518_0 ctx), ((generatedRoot.w_518_0 ctx), ((generatedRoot.w_518_0 ctx), ((generatedRoot.w_518_0 ctx), ((generatedRoot.w_520_0 ctx), ((generatedRoot.w_522_0 ctx), ((generatedRoot.w_524_0 ctx), ((generatedRoot.w_526_0 ctx), ())))))))))) ((generatedRoot.w_529_0 ctx), ((generatedRoot.w_529_1 ctx), ((generatedRoot.w_529_2 ctx), ((generatedRoot.w_529_3 ctx), ((generatedRoot.w_529_4 ctx), ((generatedRoot.w_529_5 ctx), ((generatedRoot.w_529_6 ctx), ((generatedRoot.w_529_7 ctx), ((generatedRoot.w_529_8 ctx), ((generatedRoot.w_529_9 ctx), ()))))))))))

theorem generatedRoot.facts_21  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_21 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_521_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_528_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_494_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_497_0 ctx) (generatedRoot.w_498_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_501_0 ctx) (generatedRoot.w_502_0 ctx) (generatedRoot.w_505_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_509_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_513_0 ctx) (generatedRoot.w_514_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_518_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_529_1 ctx) (generatedRoot.w_529_2 ctx) (generatedRoot.w_529_3 ctx) (generatedRoot.w_529_4 ctx) (generatedRoot.w_529_5 ctx) (generatedRoot.w_529_6 ctx) (generatedRoot.w_529_7 ctx) (generatedRoot.w_529_8 ctx) (generatedRoot.w_529_9 ctx)) :
    generatedRoot.Facts_21 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_22  (ctx : generatedRootCtx) : Prop where
  h530_0 : MxxRuntime.familyGetStatic (generatedRoot.w_527_0 ctx) (0) (generatedRoot.w_530_0 ctx)
  t705 : 0 ≤ (0)
  t706 : (0) < 1
  h532_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_531_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_532_0 ctx)
  h533_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_531_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_533_0 ctx)
  h532_0_1 : MxxRuntime.concatRows (generatedRoot.w_530_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_534_concat_1 ctx)
  h534_0 : MxxRuntime.concatRows (generatedRoot.w_534_concat_1 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx)
  h535_0 : MxxRuntime.familyGetStatic (generatedRoot.w_528_0 ctx) (0) (generatedRoot.w_535_0 ctx)
  h536_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_531_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_536_0 ctx)
  h537_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_531_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_537_0 ctx)
  h536_0_1 : MxxRuntime.concatRows (generatedRoot.w_535_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_concat_1 ctx)
  h538_0 : MxxRuntime.concatRows (generatedRoot.w_538_concat_1 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_538_0 ctx)
  h539_0 : MxxRuntime.concatColumns (generatedRoot.w_534_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_539_0 ctx)
  t717 : 0 ≤ (0)
  t718 : (0) < 1
  h540_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_56_4 ctx) (generatedRoot.w_540_0 ctx)
  t720 : 0 ≤ (0)
  t721 : (0) < 1
  h541_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_56_0 ctx) (generatedRoot.w_541_0 ctx)
  t723 : 0 ≤ (0)
  t724 : (0) < 1
  h542_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_56_2 ctx) (generatedRoot.w_542_0 ctx)
  h541_0_1 : MxxRuntime.concatRows (generatedRoot.w_540_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_543_concat_1 ctx)
  h543_0 : MxxRuntime.concatRows (generatedRoot.w_543_concat_1 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_543_0 ctx)
  t728 : 0 ≤ (0)
  t729 : (0) < 1
  h544_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_56_5 ctx) (generatedRoot.w_544_0 ctx)
  t731 : 0 ≤ (0)
  t732 : (0) < 1
  h545_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_56_1 ctx) (generatedRoot.w_545_0 ctx)
  t734 : 0 ≤ (0)
  t735 : (0) < 1

theorem generatedRoot.facts_22  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_22 hashModel (generatedRoot.w_7_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_528_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_534_concat_1 ctx) (generatedRoot.w_535_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_538_concat_1 ctx) (generatedRoot.w_539_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_543_concat_1 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx)) :
    generatedRoot.Facts_22 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_23  (ctx : generatedRootCtx) : Prop where
  h546_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_56_3 ctx) (generatedRoot.w_546_0 ctx)
  h545_0_1 : MxxRuntime.concatRows (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_concat_1 ctx)
  h547_0 : MxxRuntime.concatRows (generatedRoot.w_547_concat_1 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_547_0 ctx)
  h548_0 : MxxRuntime.concatColumns (generatedRoot.w_543_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_548_0 ctx)
  h549_0 : MxxRuntime.concatRows (generatedRoot.w_529_7 ctx) (generatedRoot.w_529_9 ctx) (generatedRoot.w_549_0 ctx)
  h550_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx)
  h551_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_550_0 ctx) (generatedRoot.w_551_0 ctx)
  t743 : 0 ≤ 1
  t744 : 1 < 2
  t745 : 2 ≤ 3
  t746 : 0 ≤ 0
  t747 : 0 < 25
  t748 : 25 ≤ 25
  h553_0 : MxxRuntime.sliceMatrix (generatedRoot.w_552_0 ctx) 1 2 0 25 (generatedRoot.w_553_0 ctx)
  t750 : 0 ≤ 2
  t751 : 2 < 3
  t752 : 3 ≤ 3
  t753 : 0 ≤ 0
  t754 : 0 < 25
  t755 : 25 ≤ 25
  h554_0 : MxxRuntime.sliceMatrix (generatedRoot.w_552_0 ctx) 2 3 0 25 (generatedRoot.w_554_0 ctx)
  h555_0 : MxxRuntime.concatRows (generatedRoot.w_553_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_555_0 ctx)
  h556_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_555_0 ctx) (generatedRoot.w_556_0 ctx)
  h557_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_556_0 ctx) (generatedRoot.w_557_0 ctx)
  t760 : 0 ≤ 0
  t761 : 0 < 1
  t762 : 1 ≤ 3
  t763 : 0 ≤ 0
  t764 : 0 < 25
  t765 : 25 ≤ 25
  h559_0 : MxxRuntime.sliceMatrix (generatedRoot.w_558_0 ctx) 0 1 0 25 (generatedRoot.w_559_0 ctx)
  t767 : 0 ≤ 0

theorem generatedRoot.facts_23  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_23 backend (generatedRoot.w_56_3 ctx) (generatedRoot.w_529_7 ctx) (generatedRoot.w_529_9 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_547_concat_1 ctx) (generatedRoot.w_548_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_555_0 ctx) (generatedRoot.w_556_0 ctx) (generatedRoot.w_557_0 ctx) (generatedRoot.w_558_0 ctx) (generatedRoot.w_559_0 ctx)) :
    generatedRoot.Facts_23 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_24  (ctx : generatedRootCtx) : Prop where
  t768 : 0 < 1
  t769 : 1 ≤ 3
  t770 : 0 ≤ 0
  t771 : 0 < 25
  t772 : 25 ≤ 25
  h560_0 : MxxRuntime.sliceMatrix (generatedRoot.w_552_0 ctx) 0 1 0 25 (generatedRoot.w_560_0 ctx)
  h563_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_563_0 ctx)
  h566_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_565_0 ctx) (generatedRoot.w_566_0 ctx)
  h567_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_566_0 ctx) (generatedRoot.w_567_0 ctx)
  h569_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_569_0 ctx)
  h572_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_571_0 ctx) (generatedRoot.w_572_0 ctx)
  h573_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_572_0 ctx) (generatedRoot.w_573_0 ctx)
  h578_0 : MxxRuntime.familyGetStatic (generatedRoot.w_577_0 ctx) (0) (generatedRoot.w_578_0 ctx)
  h580_0 : MxxRuntime.familyGetStatic (generatedRoot.w_579_0 ctx) (0) (generatedRoot.w_580_0 ctx)
  h582_0 : MxxRuntime.familyGetStatic (generatedRoot.w_581_0 ctx) (0) (generatedRoot.w_582_0 ctx)
  h584_0 : MxxRuntime.familyGetStatic (generatedRoot.w_583_0 ctx) (0) (generatedRoot.w_584_0 ctx)
  t784 : 0 ≤ (0)
  h587_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_587 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_585_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_586_0 ctx), ((generatedRoot.w_56_4 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_2 ctx), ((generatedRoot.w_56_5 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_56_3 ctx), ((generatedRoot.w_577_0 ctx), ((generatedRoot.w_579_0 ctx), ((generatedRoot.w_581_0 ctx), ((generatedRoot.w_583_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_576_0 ctx), ((generatedRoot.w_576_0 ctx), ((generatedRoot.w_576_0 ctx), ((generatedRoot.w_576_0 ctx), ((generatedRoot.w_576_0 ctx), ((generatedRoot.w_576_0 ctx), ((generatedRoot.w_578_0 ctx), ((generatedRoot.w_580_0 ctx), ((generatedRoot.w_582_0 ctx), ((generatedRoot.w_584_0 ctx), ())))))))))) ((generatedRoot.w_587_0 ctx), ((generatedRoot.w_587_1 ctx), ((generatedRoot.w_587_2 ctx), ((generatedRoot.w_587_3 ctx), ((generatedRoot.w_587_4 ctx), ((generatedRoot.w_587_5 ctx), ((generatedRoot.w_587_6 ctx), ((generatedRoot.w_587_7 ctx), ((generatedRoot.w_587_8 ctx), ((generatedRoot.w_587_9 ctx), ()))))))))))
  h588_0 : MxxRuntime.familyGetStatic (generatedRoot.w_585_0 ctx) (0) (generatedRoot.w_588_0 ctx)
  t787 : 0 ≤ (0)
  t788 : (0) < 1
  h590_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_589_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_590_0 ctx)
  h591_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_589_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_591_0 ctx)
  h590_0_1 : MxxRuntime.concatRows (generatedRoot.w_588_0 ctx) (generatedRoot.w_590_0 ctx) (generatedRoot.w_592_concat_1 ctx)
  h592_0 : MxxRuntime.concatRows (generatedRoot.w_592_concat_1 ctx) (generatedRoot.w_591_0 ctx) (generatedRoot.w_592_0 ctx)
  h593_0 : MxxRuntime.familyGetStatic (generatedRoot.w_586_0 ctx) (0) (generatedRoot.w_593_0 ctx)
  h594_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_589_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_594_0 ctx)
  h595_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_589_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_595_0 ctx)
  h594_0_1 : MxxRuntime.concatRows (generatedRoot.w_593_0 ctx) (generatedRoot.w_594_0 ctx) (generatedRoot.w_596_concat_1 ctx)
  h596_0 : MxxRuntime.concatRows (generatedRoot.w_596_concat_1 ctx) (generatedRoot.w_595_0 ctx) (generatedRoot.w_596_0 ctx)
  h597_0 : MxxRuntime.concatColumns (generatedRoot.w_592_0 ctx) (generatedRoot.w_596_0 ctx) (generatedRoot.w_597_0 ctx)
  t799 : 0 ≤ (0)

theorem generatedRoot.facts_24  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_24 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_577_0 ctx) (generatedRoot.w_579_0 ctx) (generatedRoot.w_581_0 ctx) (generatedRoot.w_583_0 ctx) (generatedRoot.w_585_0 ctx) (generatedRoot.w_586_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_560_0 ctx) (generatedRoot.w_563_0 ctx) (generatedRoot.w_565_0 ctx) (generatedRoot.w_566_0 ctx) (generatedRoot.w_567_0 ctx) (generatedRoot.w_569_0 ctx) (generatedRoot.w_571_0 ctx) (generatedRoot.w_572_0 ctx) (generatedRoot.w_573_0 ctx) (generatedRoot.w_576_0 ctx) (generatedRoot.w_578_0 ctx) (generatedRoot.w_580_0 ctx) (generatedRoot.w_582_0 ctx) (generatedRoot.w_584_0 ctx) (generatedRoot.w_587_0 ctx) (generatedRoot.w_587_1 ctx) (generatedRoot.w_587_2 ctx) (generatedRoot.w_587_3 ctx) (generatedRoot.w_587_4 ctx) (generatedRoot.w_587_5 ctx) (generatedRoot.w_587_6 ctx) (generatedRoot.w_587_7 ctx) (generatedRoot.w_587_8 ctx) (generatedRoot.w_587_9 ctx) (generatedRoot.w_588_0 ctx) (generatedRoot.w_589_0 ctx) (generatedRoot.w_590_0 ctx) (generatedRoot.w_591_0 ctx) (generatedRoot.w_592_0 ctx) (generatedRoot.w_592_concat_1 ctx) (generatedRoot.w_593_0 ctx) (generatedRoot.w_594_0 ctx) (generatedRoot.w_595_0 ctx) (generatedRoot.w_596_0 ctx) (generatedRoot.w_596_concat_1 ctx) (generatedRoot.w_597_0 ctx)) :
    generatedRoot.Facts_24 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_25  (ctx : generatedRootCtx) : Prop where
  t800 : (0) < 1
  h598_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_56_4 ctx) (generatedRoot.w_598_0 ctx)
  t802 : 0 ≤ (0)
  t803 : (0) < 1
  h599_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_56_0 ctx) (generatedRoot.w_599_0 ctx)
  t805 : 0 ≤ (0)
  t806 : (0) < 1
  h600_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_56_2 ctx) (generatedRoot.w_600_0 ctx)
  h599_0_1 : MxxRuntime.concatRows (generatedRoot.w_598_0 ctx) (generatedRoot.w_599_0 ctx) (generatedRoot.w_601_concat_1 ctx)
  h601_0 : MxxRuntime.concatRows (generatedRoot.w_601_concat_1 ctx) (generatedRoot.w_600_0 ctx) (generatedRoot.w_601_0 ctx)
  t810 : 0 ≤ (0)
  t811 : (0) < 1
  h602_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_56_5 ctx) (generatedRoot.w_602_0 ctx)
  t813 : 0 ≤ (0)
  t814 : (0) < 1
  h603_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_56_1 ctx) (generatedRoot.w_603_0 ctx)
  t816 : 0 ≤ (0)
  t817 : (0) < 1
  h604_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_56_3 ctx) (generatedRoot.w_604_0 ctx)
  h603_0_1 : MxxRuntime.concatRows (generatedRoot.w_602_0 ctx) (generatedRoot.w_603_0 ctx) (generatedRoot.w_605_concat_1 ctx)
  h605_0 : MxxRuntime.concatRows (generatedRoot.w_605_concat_1 ctx) (generatedRoot.w_604_0 ctx) (generatedRoot.w_605_0 ctx)
  h606_0 : MxxRuntime.concatColumns (generatedRoot.w_601_0 ctx) (generatedRoot.w_605_0 ctx) (generatedRoot.w_606_0 ctx)
  h607_0 : MxxRuntime.concatRows (generatedRoot.w_587_7 ctx) (generatedRoot.w_587_9 ctx) (generatedRoot.w_607_0 ctx)
  h608_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_607_0 ctx) (generatedRoot.w_608_0 ctx)
  h609_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_608_0 ctx) (generatedRoot.w_609_0 ctx)
  t825 : 0 ≤ 1
  t826 : 1 < 2
  t827 : 2 ≤ 3
  t828 : 0 ≤ 0
  t829 : 0 < 25
  t830 : 25 ≤ 25
  h611_0 : MxxRuntime.sliceMatrix (generatedRoot.w_610_0 ctx) 1 2 0 25 (generatedRoot.w_611_0 ctx)

theorem generatedRoot.facts_25  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_25 backend (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_587_7 ctx) (generatedRoot.w_587_9 ctx) (generatedRoot.w_598_0 ctx) (generatedRoot.w_599_0 ctx) (generatedRoot.w_600_0 ctx) (generatedRoot.w_601_0 ctx) (generatedRoot.w_601_concat_1 ctx) (generatedRoot.w_602_0 ctx) (generatedRoot.w_603_0 ctx) (generatedRoot.w_604_0 ctx) (generatedRoot.w_605_0 ctx) (generatedRoot.w_605_concat_1 ctx) (generatedRoot.w_606_0 ctx) (generatedRoot.w_607_0 ctx) (generatedRoot.w_608_0 ctx) (generatedRoot.w_609_0 ctx) (generatedRoot.w_610_0 ctx) (generatedRoot.w_611_0 ctx)) :
    generatedRoot.Facts_25 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_26  (ctx : generatedRootCtx) : Prop where
  t832 : 0 ≤ 2
  t833 : 2 < 3
  t834 : 3 ≤ 3
  t835 : 0 ≤ 0
  t836 : 0 < 25
  t837 : 25 ≤ 25
  h612_0 : MxxRuntime.sliceMatrix (generatedRoot.w_610_0 ctx) 2 3 0 25 (generatedRoot.w_612_0 ctx)
  h613_0 : MxxRuntime.concatRows (generatedRoot.w_611_0 ctx) (generatedRoot.w_612_0 ctx) (generatedRoot.w_613_0 ctx)
  h614_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_613_0 ctx) (generatedRoot.w_614_0 ctx)
  h615_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_614_0 ctx) (generatedRoot.w_615_0 ctx)
  t842 : 0 ≤ 0
  t843 : 0 < 1
  t844 : 1 ≤ 3
  t845 : 0 ≤ 0
  t846 : 0 < 25
  t847 : 25 ≤ 25
  h617_0 : MxxRuntime.sliceMatrix (generatedRoot.w_616_0 ctx) 0 1 0 25 (generatedRoot.w_617_0 ctx)
  t849 : 0 ≤ 0
  t850 : 0 < 1
  t851 : 1 ≤ 3
  t852 : 0 ≤ 0
  t853 : 0 < 25
  t854 : 25 ≤ 25
  h618_0 : MxxRuntime.sliceMatrix (generatedRoot.w_610_0 ctx) 0 1 0 25 (generatedRoot.w_618_0 ctx)
  h621_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_57_0 ctx) (generatedRoot.w_621_0 ctx)
  h624_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_623_0 ctx) (generatedRoot.w_624_0 ctx)
  h625_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_624_0 ctx) (generatedRoot.w_625_0 ctx)
  h627_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_627_0 ctx)
  h630_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_629_0 ctx) (generatedRoot.w_630_0 ctx)
  h631_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_630_0 ctx) (generatedRoot.w_631_0 ctx)
  h635_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_80_0 ctx), (generatedRoot.w_81_0 ctx), (generatedRoot.w_87_0 ctx), (generatedRoot.w_88_0 ctx), (generatedRoot.w_199_0 ctx), (generatedRoot.w_214_0 ctx), (generatedRoot.w_226_0 ctx), (generatedRoot.w_227_0 ctx), (generatedRoot.w_284_0 ctx), (generatedRoot.w_285_0 ctx), (generatedRoot.w_342_0 ctx), (generatedRoot.w_343_0 ctx), (generatedRoot.w_400_0 ctx), (generatedRoot.w_401_0 ctx), (generatedRoot.w_458_0 ctx), (generatedRoot.w_459_0 ctx), (generatedRoot.w_516_0 ctx), (generatedRoot.w_517_0 ctx), (generatedRoot.w_574_0 ctx), (generatedRoot.w_575_0 ctx), (generatedRoot.w_632_0 ctx), (generatedRoot.w_633_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx)] (generatedRoot.w_635_0 ctx)
  h636_1 : MxxRuntime.sha256BitColumns [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 (generatedRoot.w_636_0 ctx) (generatedRoot.w_636_1 ctx)

theorem generatedRoot.facts_26  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_26 backend (generatedRoot.w_57_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_401_0 ctx) (generatedRoot.w_458_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_517_0 ctx) (generatedRoot.w_574_0 ctx) (generatedRoot.w_575_0 ctx) (generatedRoot.w_610_0 ctx) (generatedRoot.w_611_0 ctx) (generatedRoot.w_612_0 ctx) (generatedRoot.w_613_0 ctx) (generatedRoot.w_614_0 ctx) (generatedRoot.w_615_0 ctx) (generatedRoot.w_616_0 ctx) (generatedRoot.w_617_0 ctx) (generatedRoot.w_618_0 ctx) (generatedRoot.w_621_0 ctx) (generatedRoot.w_623_0 ctx) (generatedRoot.w_624_0 ctx) (generatedRoot.w_625_0 ctx) (generatedRoot.w_627_0 ctx) (generatedRoot.w_629_0 ctx) (generatedRoot.w_630_0 ctx) (generatedRoot.w_631_0 ctx) (generatedRoot.w_632_0 ctx) (generatedRoot.w_633_0 ctx) (generatedRoot.w_634_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_636_0 ctx) (generatedRoot.w_636_1 ctx)) :
    generatedRoot.Facts_26 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_27  (ctx : generatedRootCtx) : Prop where
  h638_0 : MxxRuntime.familyGetStatic (generatedRoot.w_637_0 ctx) (1) (generatedRoot.w_638_0 ctx)
  h641_0 : MxxRuntime.familyGetStatic (generatedRoot.w_640_0 ctx) (1) (generatedRoot.w_641_0 ctx)
  h643_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_643_0 ctx)
  h644_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_644_0 ctx)
  h645_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_645_0 ctx)
  h646_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_646_0 ctx)
  h649_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_649_0 ctx)
  h654_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_654_0 ctx)
  h657_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_657_0 ctx)
  t873 : (5) = 5
  h660_5 : (∀ i : Fin 5, parallel_generatedRoot_660 backend hashModel params i ((generatedRoot.w_636_1 ctx), ((generatedRoot.w_638_0 ctx), ((generatedRoot.w_639_0 ctx), ((generatedRoot.w_636_0 ctx), ((generatedRoot.w_641_0 ctx), ((generatedRoot.w_642_0 ctx), ((generatedRoot.w_643_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_644_0 ctx), ((generatedRoot.w_645_0 ctx), ((generatedRoot.w_646_0 ctx), ((generatedRoot.w_647_0 ctx), ((generatedRoot.w_648_0 ctx), ((generatedRoot.w_649_0 ctx), ((generatedRoot.w_651_0 ctx), ((generatedRoot.w_653_0 ctx), ((generatedRoot.w_87_0 ctx), ((generatedRoot.w_656_0 ctx), ((generatedRoot.w_659_0 ctx), ((generatedRoot.w_88_0 ctx), ())))))))))))))))))))) (((generatedRoot.w_660_0 ctx) i), (((generatedRoot.w_660_1 ctx) i), (((generatedRoot.w_660_2 ctx) i), (((generatedRoot.w_660_3 ctx) i), (((generatedRoot.w_660_4 ctx) i), (((generatedRoot.w_660_5 ctx) i), ())))))))
  t875 : 0 ≤ 1
  t876 : 1 < 2
  t877 : 2 ≤ 3
  t878 : 0 ≤ 0
  t879 : 0 < 25
  t880 : 25 ≤ 25
  h663_0 : MxxRuntime.sliceMatrix (generatedRoot.w_189_0 ctx) 1 2 0 25 (generatedRoot.w_663_0 ctx)
  t882 : 0 ≤ 1
  t883 : 1 < 2
  t884 : 2 ≤ 3
  t885 : 0 ≤ 0
  t886 : 0 < 25
  t887 : 25 ≤ 25
  h666_0 : MxxRuntime.sliceMatrix (generatedRoot.w_209_0 ctx) 1 2 0 25 (generatedRoot.w_666_0 ctx)
  t889 : 0 ≤ 1
  t890 : 1 < 2
  t891 : 2 ≤ 3
  t892 : 0 ≤ 0
  t893 : 0 < 25
  t894 : 25 ≤ 25
  h672_0 : MxxRuntime.sliceMatrix (generatedRoot.w_268_0 ctx) 1 2 0 25 (generatedRoot.w_672_0 ctx)

theorem generatedRoot.facts_27  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_27 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_637_0 ctx) (generatedRoot.w_639_0 ctx) (generatedRoot.w_640_0 ctx) (generatedRoot.w_642_0 ctx) (generatedRoot.w_648_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_636_0 ctx) (generatedRoot.w_636_1 ctx) (generatedRoot.w_638_0 ctx) (generatedRoot.w_641_0 ctx) (generatedRoot.w_643_0 ctx) (generatedRoot.w_644_0 ctx) (generatedRoot.w_645_0 ctx) (generatedRoot.w_646_0 ctx) (generatedRoot.w_647_0 ctx) (generatedRoot.w_649_0 ctx) (generatedRoot.w_651_0 ctx) (generatedRoot.w_653_0 ctx) (generatedRoot.w_654_0 ctx) (generatedRoot.w_656_0 ctx) (generatedRoot.w_657_0 ctx) (generatedRoot.w_659_0 ctx) (generatedRoot.w_660_0 ctx) (generatedRoot.w_660_1 ctx) (generatedRoot.w_660_2 ctx) (generatedRoot.w_660_3 ctx) (generatedRoot.w_660_4 ctx) (generatedRoot.w_660_5 ctx) (generatedRoot.w_663_0 ctx) (generatedRoot.w_666_0 ctx) (generatedRoot.w_672_0 ctx)) :
    generatedRoot.Facts_27 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_28  (ctx : generatedRootCtx) : Prop where
  t896 : 0 ≤ 1
  t897 : 1 < 2
  t898 : 2 ≤ 3
  t899 : 0 ≤ 0
  t900 : 0 < 25
  t901 : 25 ≤ 25
  h677_0 : MxxRuntime.sliceMatrix (generatedRoot.w_326_0 ctx) 1 2 0 25 (generatedRoot.w_677_0 ctx)
  t903 : 0 ≤ 1
  t904 : 1 < 2
  t905 : 2 ≤ 3
  t906 : 0 ≤ 0
  t907 : 0 < 25
  t908 : 25 ≤ 25
  h682_0 : MxxRuntime.sliceMatrix (generatedRoot.w_384_0 ctx) 1 2 0 25 (generatedRoot.w_682_0 ctx)
  t910 : 0 ≤ 1
  t911 : 1 < 2
  t912 : 2 ≤ 3
  t913 : 0 ≤ 0
  t914 : 0 < 25
  t915 : 25 ≤ 25
  h687_0 : MxxRuntime.sliceMatrix (generatedRoot.w_442_0 ctx) 1 2 0 25 (generatedRoot.w_687_0 ctx)
  t917 : 0 ≤ 1
  t918 : 1 < 2
  t919 : 2 ≤ 3
  t920 : 0 ≤ 0
  t921 : 0 < 25
  t922 : 25 ≤ 25
  h692_0 : MxxRuntime.sliceMatrix (generatedRoot.w_500_0 ctx) 1 2 0 25 (generatedRoot.w_692_0 ctx)
  t924 : 0 ≤ 1
  t925 : 1 < 2
  t926 : 2 ≤ 3
  t927 : 0 ≤ 0

theorem generatedRoot.facts_28  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_28 (generatedRoot.w_326_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_442_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_677_0 ctx) (generatedRoot.w_682_0 ctx) (generatedRoot.w_687_0 ctx) (generatedRoot.w_692_0 ctx)) :
    generatedRoot.Facts_28 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_29  (ctx : generatedRootCtx) : Prop where
  t928 : 0 < 25
  t929 : 25 ≤ 25
  h697_0 : MxxRuntime.sliceMatrix (generatedRoot.w_558_0 ctx) 1 2 0 25 (generatedRoot.w_697_0 ctx)
  t931 : 0 ≤ 1
  t932 : 1 < 2
  t933 : 2 ≤ 3
  t934 : 0 ≤ 0
  t935 : 0 < 25
  t936 : 25 ≤ 25
  h702_0 : MxxRuntime.sliceMatrix (generatedRoot.w_616_0 ctx) 1 2 0 25 (generatedRoot.w_702_0 ctx)
  h707_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_661_0 ctx), (generatedRoot.w_662_0 ctx), (generatedRoot.w_651_0 ctx), (generatedRoot.w_656_0 ctx), (generatedRoot.w_665_0 ctx), (generatedRoot.w_668_0 ctx), (generatedRoot.w_670_0 ctx), (generatedRoot.w_671_0 ctx), (generatedRoot.w_675_0 ctx), (generatedRoot.w_676_0 ctx), (generatedRoot.w_680_0 ctx), (generatedRoot.w_681_0 ctx), (generatedRoot.w_685_0 ctx), (generatedRoot.w_686_0 ctx), (generatedRoot.w_690_0 ctx), (generatedRoot.w_691_0 ctx), (generatedRoot.w_695_0 ctx), (generatedRoot.w_696_0 ctx), (generatedRoot.w_700_0 ctx), (generatedRoot.w_701_0 ctx), (generatedRoot.w_705_0 ctx), (generatedRoot.w_706_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx)] (generatedRoot.w_707_0 ctx)
  t939 : 0 ≤ 2
  t940 : 2 < 3
  t941 : 3 ≤ 3
  t942 : 0 ≤ 0
  t943 : 0 < 25
  t944 : 25 ≤ 25
  h710_0 : MxxRuntime.sliceMatrix (generatedRoot.w_189_0 ctx) 2 3 0 25 (generatedRoot.w_710_0 ctx)
  t946 : 0 ≤ 2
  t947 : 2 < 3
  t948 : 3 ≤ 3
  t949 : 0 ≤ 0
  t950 : 0 < 25
  t951 : 25 ≤ 25
  h713_0 : MxxRuntime.sliceMatrix (generatedRoot.w_209_0 ctx) 2 3 0 25 (generatedRoot.w_713_0 ctx)
  t953 : 0 ≤ 2
  t954 : 2 < 3
  t955 : 3 ≤ 3
  t956 : 0 ≤ 0
  t957 : 0 < 25
  t958 : 25 ≤ 25
  h719_0 : MxxRuntime.sliceMatrix (generatedRoot.w_268_0 ctx) 2 3 0 25 (generatedRoot.w_719_0 ctx)

theorem generatedRoot.facts_29  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_29 (generatedRoot.w_189_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_558_0 ctx) (generatedRoot.w_616_0 ctx) (generatedRoot.w_634_0 ctx) (generatedRoot.w_651_0 ctx) (generatedRoot.w_656_0 ctx) (generatedRoot.w_661_0 ctx) (generatedRoot.w_662_0 ctx) (generatedRoot.w_665_0 ctx) (generatedRoot.w_668_0 ctx) (generatedRoot.w_670_0 ctx) (generatedRoot.w_671_0 ctx) (generatedRoot.w_675_0 ctx) (generatedRoot.w_676_0 ctx) (generatedRoot.w_680_0 ctx) (generatedRoot.w_681_0 ctx) (generatedRoot.w_685_0 ctx) (generatedRoot.w_686_0 ctx) (generatedRoot.w_690_0 ctx) (generatedRoot.w_691_0 ctx) (generatedRoot.w_695_0 ctx) (generatedRoot.w_696_0 ctx) (generatedRoot.w_697_0 ctx) (generatedRoot.w_700_0 ctx) (generatedRoot.w_701_0 ctx) (generatedRoot.w_702_0 ctx) (generatedRoot.w_705_0 ctx) (generatedRoot.w_706_0 ctx) (generatedRoot.w_707_0 ctx) (generatedRoot.w_710_0 ctx) (generatedRoot.w_713_0 ctx) (generatedRoot.w_719_0 ctx)) :
    generatedRoot.Facts_29 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_30  (ctx : generatedRootCtx) : Prop where
  t960 : 0 ≤ 2
  t961 : 2 < 3
  t962 : 3 ≤ 3
  t963 : 0 ≤ 0
  t964 : 0 < 25
  t965 : 25 ≤ 25
  h724_0 : MxxRuntime.sliceMatrix (generatedRoot.w_326_0 ctx) 2 3 0 25 (generatedRoot.w_724_0 ctx)
  t967 : 0 ≤ 2
  t968 : 2 < 3
  t969 : 3 ≤ 3
  t970 : 0 ≤ 0
  t971 : 0 < 25
  t972 : 25 ≤ 25
  h729_0 : MxxRuntime.sliceMatrix (generatedRoot.w_384_0 ctx) 2 3 0 25 (generatedRoot.w_729_0 ctx)
  t974 : 0 ≤ 2
  t975 : 2 < 3
  t976 : 3 ≤ 3
  t977 : 0 ≤ 0
  t978 : 0 < 25
  t979 : 25 ≤ 25
  h734_0 : MxxRuntime.sliceMatrix (generatedRoot.w_442_0 ctx) 2 3 0 25 (generatedRoot.w_734_0 ctx)
  t981 : 0 ≤ 2
  t982 : 2 < 3
  t983 : 3 ≤ 3
  t984 : 0 ≤ 0
  t985 : 0 < 25
  t986 : 25 ≤ 25
  h739_0 : MxxRuntime.sliceMatrix (generatedRoot.w_500_0 ctx) 2 3 0 25 (generatedRoot.w_739_0 ctx)
  t988 : 0 ≤ 2
  t989 : 2 < 3
  t990 : 3 ≤ 3
  t991 : 0 ≤ 0

theorem generatedRoot.facts_30  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_30 (generatedRoot.w_326_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_442_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_724_0 ctx) (generatedRoot.w_729_0 ctx) (generatedRoot.w_734_0 ctx) (generatedRoot.w_739_0 ctx)) :
    generatedRoot.Facts_30 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_31  (ctx : generatedRootCtx) : Prop where
  t992 : 0 < 25
  t993 : 25 ≤ 25
  h744_0 : MxxRuntime.sliceMatrix (generatedRoot.w_558_0 ctx) 2 3 0 25 (generatedRoot.w_744_0 ctx)
  t995 : 0 ≤ 2
  t996 : 2 < 3
  t997 : 3 ≤ 3
  t998 : 0 ≤ 0
  t999 : 0 < 25
  t1000 : 25 ≤ 25
  h749_0 : MxxRuntime.sliceMatrix (generatedRoot.w_616_0 ctx) 2 3 0 25 (generatedRoot.w_749_0 ctx)
  h754_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_708_0 ctx), (generatedRoot.w_709_0 ctx), (generatedRoot.w_653_0 ctx), (generatedRoot.w_659_0 ctx), (generatedRoot.w_712_0 ctx), (generatedRoot.w_715_0 ctx), (generatedRoot.w_717_0 ctx), (generatedRoot.w_718_0 ctx), (generatedRoot.w_722_0 ctx), (generatedRoot.w_723_0 ctx), (generatedRoot.w_727_0 ctx), (generatedRoot.w_728_0 ctx), (generatedRoot.w_732_0 ctx), (generatedRoot.w_733_0 ctx), (generatedRoot.w_737_0 ctx), (generatedRoot.w_738_0 ctx), (generatedRoot.w_742_0 ctx), (generatedRoot.w_743_0 ctx), (generatedRoot.w_747_0 ctx), (generatedRoot.w_748_0 ctx), (generatedRoot.w_752_0 ctx), (generatedRoot.w_753_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_634_0 ctx)] (generatedRoot.w_754_0 ctx)
  h756_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_756_0 ctx)
  h757_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_757_0 ctx)
  h758_0 : MxxRuntime.concatRows (generatedRoot.w_756_0 ctx) (generatedRoot.w_757_0 ctx) (generatedRoot.w_758_0 ctx)
  h759_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_758_0 ctx) (generatedRoot.w_759_0 ctx)
  h760_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_760_0 ctx)
  h761_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_761_0 ctx)
  h762_0 : MxxRuntime.concatRows (generatedRoot.w_760_0 ctx) (generatedRoot.w_761_0 ctx) (generatedRoot.w_762_0 ctx)
  h763_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_762_0 ctx) (generatedRoot.w_763_0 ctx)
  h764_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_764_0 ctx)
  h765_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_765_0 ctx)
  h766_0 : MxxRuntime.concatRows (generatedRoot.w_764_0 ctx) (generatedRoot.w_765_0 ctx) (generatedRoot.w_766_0 ctx)
  h767_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_766_0 ctx) (generatedRoot.w_767_0 ctx)
  h768_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_768_0 ctx)
  h769_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_769_0 ctx)
  h770_0 : MxxRuntime.concatRows (generatedRoot.w_768_0 ctx) (generatedRoot.w_769_0 ctx) (generatedRoot.w_770_0 ctx)
  h771_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_770_0 ctx) (generatedRoot.w_771_0 ctx)
  h772_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_772_0 ctx)
  h773_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_773_0 ctx)
  h774_0 : MxxRuntime.concatRows (generatedRoot.w_772_0 ctx) (generatedRoot.w_773_0 ctx) (generatedRoot.w_774_0 ctx)
  h775_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_774_0 ctx) (generatedRoot.w_775_0 ctx)
  h776_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_776_0 ctx)

theorem generatedRoot.facts_31  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_31 hashModel (generatedRoot.w_755_0 ctx) (generatedRoot.w_558_0 ctx) (generatedRoot.w_616_0 ctx) (generatedRoot.w_634_0 ctx) (generatedRoot.w_653_0 ctx) (generatedRoot.w_659_0 ctx) (generatedRoot.w_708_0 ctx) (generatedRoot.w_709_0 ctx) (generatedRoot.w_712_0 ctx) (generatedRoot.w_715_0 ctx) (generatedRoot.w_717_0 ctx) (generatedRoot.w_718_0 ctx) (generatedRoot.w_722_0 ctx) (generatedRoot.w_723_0 ctx) (generatedRoot.w_727_0 ctx) (generatedRoot.w_728_0 ctx) (generatedRoot.w_732_0 ctx) (generatedRoot.w_733_0 ctx) (generatedRoot.w_737_0 ctx) (generatedRoot.w_738_0 ctx) (generatedRoot.w_742_0 ctx) (generatedRoot.w_743_0 ctx) (generatedRoot.w_744_0 ctx) (generatedRoot.w_747_0 ctx) (generatedRoot.w_748_0 ctx) (generatedRoot.w_749_0 ctx) (generatedRoot.w_752_0 ctx) (generatedRoot.w_753_0 ctx) (generatedRoot.w_754_0 ctx) (generatedRoot.w_756_0 ctx) (generatedRoot.w_757_0 ctx) (generatedRoot.w_758_0 ctx) (generatedRoot.w_759_0 ctx) (generatedRoot.w_760_0 ctx) (generatedRoot.w_761_0 ctx) (generatedRoot.w_762_0 ctx) (generatedRoot.w_763_0 ctx) (generatedRoot.w_764_0 ctx) (generatedRoot.w_765_0 ctx) (generatedRoot.w_766_0 ctx) (generatedRoot.w_767_0 ctx) (generatedRoot.w_768_0 ctx) (generatedRoot.w_769_0 ctx) (generatedRoot.w_770_0 ctx) (generatedRoot.w_771_0 ctx) (generatedRoot.w_772_0 ctx) (generatedRoot.w_773_0 ctx) (generatedRoot.w_774_0 ctx) (generatedRoot.w_775_0 ctx) (generatedRoot.w_776_0 ctx)) :
    generatedRoot.Facts_31 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_32  (ctx : generatedRootCtx) : Prop where
  h777_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_777_0 ctx)
  h778_0 : MxxRuntime.concatRows (generatedRoot.w_776_0 ctx) (generatedRoot.w_777_0 ctx) (generatedRoot.w_778_0 ctx)
  h779_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_778_0 ctx) (generatedRoot.w_779_0 ctx)
  h781_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_780_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_781_0 ctx)
  h782_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_780_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_782_0 ctx)
  h783_0 : MxxRuntime.concatRows (generatedRoot.w_781_0 ctx) (generatedRoot.w_782_0 ctx) (generatedRoot.w_783_0 ctx)
  h784_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_783_0 ctx) (generatedRoot.w_784_0 ctx)
  h786_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_785_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_786_0 ctx)
  h787_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_785_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_787_0 ctx)
  h788_0 : MxxRuntime.concatRows (generatedRoot.w_786_0 ctx) (generatedRoot.w_787_0 ctx) (generatedRoot.w_788_0 ctx)
  h789_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_788_0 ctx) (generatedRoot.w_789_0 ctx)
  h791_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_790_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_791_0 ctx)
  h792_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_790_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_792_0 ctx)
  h793_0 : MxxRuntime.concatRows (generatedRoot.w_791_0 ctx) (generatedRoot.w_792_0 ctx) (generatedRoot.w_793_0 ctx)
  h794_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_793_0 ctx) (generatedRoot.w_794_0 ctx)
  h796_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_795_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_796_0 ctx)
  h797_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_795_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_797_0 ctx)
  h798_0 : MxxRuntime.concatRows (generatedRoot.w_796_0 ctx) (generatedRoot.w_797_0 ctx) (generatedRoot.w_798_0 ctx)
  h799_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_798_0 ctx) (generatedRoot.w_799_0 ctx)
  h801_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_800_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_801_0 ctx)
  h802_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_800_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_802_0 ctx)
  h803_0 : MxxRuntime.concatRows (generatedRoot.w_801_0 ctx) (generatedRoot.w_802_0 ctx) (generatedRoot.w_803_0 ctx)
  h804_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_803_0 ctx) (generatedRoot.w_804_0 ctx)
  h806_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_805_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_806_0 ctx)
  h807_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_805_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_807_0 ctx)
  h808_0 : MxxRuntime.concatRows (generatedRoot.w_806_0 ctx) (generatedRoot.w_807_0 ctx) (generatedRoot.w_808_0 ctx)
  h809_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_808_0 ctx) (generatedRoot.w_809_0 ctx)
  h811_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_810_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_811_0 ctx)
  h812_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_810_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_812_0 ctx)
  h813_0 : MxxRuntime.concatRows (generatedRoot.w_811_0 ctx) (generatedRoot.w_812_0 ctx) (generatedRoot.w_813_0 ctx)
  h814_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_813_0 ctx) (generatedRoot.w_814_0 ctx)
  h816_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_815_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_816_0 ctx)

theorem generatedRoot.facts_32  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_32 hashModel (generatedRoot.w_755_0 ctx) (generatedRoot.w_776_0 ctx) (generatedRoot.w_777_0 ctx) (generatedRoot.w_778_0 ctx) (generatedRoot.w_779_0 ctx) (generatedRoot.w_780_0 ctx) (generatedRoot.w_781_0 ctx) (generatedRoot.w_782_0 ctx) (generatedRoot.w_783_0 ctx) (generatedRoot.w_784_0 ctx) (generatedRoot.w_785_0 ctx) (generatedRoot.w_786_0 ctx) (generatedRoot.w_787_0 ctx) (generatedRoot.w_788_0 ctx) (generatedRoot.w_789_0 ctx) (generatedRoot.w_790_0 ctx) (generatedRoot.w_791_0 ctx) (generatedRoot.w_792_0 ctx) (generatedRoot.w_793_0 ctx) (generatedRoot.w_794_0 ctx) (generatedRoot.w_795_0 ctx) (generatedRoot.w_796_0 ctx) (generatedRoot.w_797_0 ctx) (generatedRoot.w_798_0 ctx) (generatedRoot.w_799_0 ctx) (generatedRoot.w_800_0 ctx) (generatedRoot.w_801_0 ctx) (generatedRoot.w_802_0 ctx) (generatedRoot.w_803_0 ctx) (generatedRoot.w_804_0 ctx) (generatedRoot.w_805_0 ctx) (generatedRoot.w_806_0 ctx) (generatedRoot.w_807_0 ctx) (generatedRoot.w_808_0 ctx) (generatedRoot.w_809_0 ctx) (generatedRoot.w_810_0 ctx) (generatedRoot.w_811_0 ctx) (generatedRoot.w_812_0 ctx) (generatedRoot.w_813_0 ctx) (generatedRoot.w_814_0 ctx) (generatedRoot.w_815_0 ctx) (generatedRoot.w_816_0 ctx)) :
    generatedRoot.Facts_32 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_33  (ctx : generatedRootCtx) : Prop where
  h817_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_815_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_817_0 ctx)
  h818_0 : MxxRuntime.concatRows (generatedRoot.w_816_0 ctx) (generatedRoot.w_817_0 ctx) (generatedRoot.w_818_0 ctx)
  h819_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_818_0 ctx) (generatedRoot.w_819_0 ctx)
  h821_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_820_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_821_0 ctx)
  h822_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_820_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_822_0 ctx)
  h823_0 : MxxRuntime.concatRows (generatedRoot.w_821_0 ctx) (generatedRoot.w_822_0 ctx) (generatedRoot.w_823_0 ctx)
  h824_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_823_0 ctx) (generatedRoot.w_824_0 ctx)
  h826_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_825_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_826_0 ctx)
  h827_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_825_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_827_0 ctx)
  h828_0 : MxxRuntime.concatRows (generatedRoot.w_826_0 ctx) (generatedRoot.w_827_0 ctx) (generatedRoot.w_828_0 ctx)
  h829_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_828_0 ctx) (generatedRoot.w_829_0 ctx)
  h831_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_830_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_831_0 ctx)
  h832_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_830_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_832_0 ctx)
  h833_0 : MxxRuntime.concatRows (generatedRoot.w_831_0 ctx) (generatedRoot.w_832_0 ctx) (generatedRoot.w_833_0 ctx)
  h834_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_833_0 ctx) (generatedRoot.w_834_0 ctx)
  h836_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_835_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_836_0 ctx)
  h837_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_835_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_837_0 ctx)
  h838_0 : MxxRuntime.concatRows (generatedRoot.w_836_0 ctx) (generatedRoot.w_837_0 ctx) (generatedRoot.w_838_0 ctx)
  h839_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_838_0 ctx) (generatedRoot.w_839_0 ctx)
  h841_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_840_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_841_0 ctx)
  h842_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_840_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_842_0 ctx)
  h843_0 : MxxRuntime.concatRows (generatedRoot.w_841_0 ctx) (generatedRoot.w_842_0 ctx) (generatedRoot.w_843_0 ctx)
  h844_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_843_0 ctx) (generatedRoot.w_844_0 ctx)
  h846_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_845_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_846_0 ctx)
  h847_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_845_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_847_0 ctx)
  h848_0 : MxxRuntime.concatRows (generatedRoot.w_846_0 ctx) (generatedRoot.w_847_0 ctx) (generatedRoot.w_848_0 ctx)
  h849_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_848_0 ctx) (generatedRoot.w_849_0 ctx)
  h851_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_850_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_851_0 ctx)
  h852_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_850_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_852_0 ctx)
  h853_0 : MxxRuntime.concatRows (generatedRoot.w_851_0 ctx) (generatedRoot.w_852_0 ctx) (generatedRoot.w_853_0 ctx)
  h854_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_853_0 ctx) (generatedRoot.w_854_0 ctx)
  h856_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_855_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_856_0 ctx)

theorem generatedRoot.facts_33  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_33 hashModel (generatedRoot.w_755_0 ctx) (generatedRoot.w_815_0 ctx) (generatedRoot.w_816_0 ctx) (generatedRoot.w_817_0 ctx) (generatedRoot.w_818_0 ctx) (generatedRoot.w_819_0 ctx) (generatedRoot.w_820_0 ctx) (generatedRoot.w_821_0 ctx) (generatedRoot.w_822_0 ctx) (generatedRoot.w_823_0 ctx) (generatedRoot.w_824_0 ctx) (generatedRoot.w_825_0 ctx) (generatedRoot.w_826_0 ctx) (generatedRoot.w_827_0 ctx) (generatedRoot.w_828_0 ctx) (generatedRoot.w_829_0 ctx) (generatedRoot.w_830_0 ctx) (generatedRoot.w_831_0 ctx) (generatedRoot.w_832_0 ctx) (generatedRoot.w_833_0 ctx) (generatedRoot.w_834_0 ctx) (generatedRoot.w_835_0 ctx) (generatedRoot.w_836_0 ctx) (generatedRoot.w_837_0 ctx) (generatedRoot.w_838_0 ctx) (generatedRoot.w_839_0 ctx) (generatedRoot.w_840_0 ctx) (generatedRoot.w_841_0 ctx) (generatedRoot.w_842_0 ctx) (generatedRoot.w_843_0 ctx) (generatedRoot.w_844_0 ctx) (generatedRoot.w_845_0 ctx) (generatedRoot.w_846_0 ctx) (generatedRoot.w_847_0 ctx) (generatedRoot.w_848_0 ctx) (generatedRoot.w_849_0 ctx) (generatedRoot.w_850_0 ctx) (generatedRoot.w_851_0 ctx) (generatedRoot.w_852_0 ctx) (generatedRoot.w_853_0 ctx) (generatedRoot.w_854_0 ctx) (generatedRoot.w_855_0 ctx) (generatedRoot.w_856_0 ctx)) :
    generatedRoot.Facts_33 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_34  (ctx : generatedRootCtx) : Prop where
  h857_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_855_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_857_0 ctx)
  h858_0 : MxxRuntime.concatRows (generatedRoot.w_856_0 ctx) (generatedRoot.w_857_0 ctx) (generatedRoot.w_858_0 ctx)
  h859_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_858_0 ctx) (generatedRoot.w_859_0 ctx)
  h861_0 : MxxRuntime.hashSample (hashModel) ([100, 105, 111, 50, 47, 97, 110, 99, 104, 111, 114, 45, 112, 97, 100, 100, 105, 110, 103]) ([]) ((generatedRoot.w_755_0 ctx)) (generatedRoot.w_861_0 ctx)
  h862_0 : MxxRuntime.concatColumns (generatedRoot.w_860_0 ctx) (generatedRoot.w_861_0 ctx) (generatedRoot.w_862_0 ctx)
  t1093 : 0 ≤ 0
  t1094 : 0 < 2
  t1095 : 2 ≤ 2
  t1096 : 0 ≤ 0
  t1097 : 0 < 25
  t1098 : 25 ≤ 100
  h863_0 : MxxRuntime.sliceMatrix (generatedRoot.w_862_0 ctx) 0 2 0 25 (generatedRoot.w_863_0 ctx)
  h864_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_863_0 ctx) (generatedRoot.w_864_0 ctx)
  t1101 : 0 ≤ 0
  t1102 : 0 < 2
  t1103 : 2 ≤ 2
  t1104 : 0 ≤ 25
  t1105 : 25 < 50
  t1106 : 50 ≤ 100
  h865_0 : MxxRuntime.sliceMatrix (generatedRoot.w_862_0 ctx) 0 2 25 50 (generatedRoot.w_865_0 ctx)
  h866_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_865_0 ctx) (generatedRoot.w_866_0 ctx)
  t1109 : 0 ≤ 0
  t1110 : 0 < 2
  t1111 : 2 ≤ 2
  t1112 : 0 ≤ 50
  t1113 : 50 < 75
  t1114 : 75 ≤ 100
  h867_0 : MxxRuntime.sliceMatrix (generatedRoot.w_862_0 ctx) 0 2 50 75 (generatedRoot.w_867_0 ctx)
  h868_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_867_0 ctx) (generatedRoot.w_868_0 ctx)
  t1117 : 0 ≤ 0
  t1118 : 0 < 2
  t1119 : 2 ≤ 2

theorem generatedRoot.facts_34  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_34 hashModel (generatedRoot.w_755_0 ctx) (generatedRoot.w_860_0 ctx) (generatedRoot.w_855_0 ctx) (generatedRoot.w_856_0 ctx) (generatedRoot.w_857_0 ctx) (generatedRoot.w_858_0 ctx) (generatedRoot.w_859_0 ctx) (generatedRoot.w_861_0 ctx) (generatedRoot.w_862_0 ctx) (generatedRoot.w_863_0 ctx) (generatedRoot.w_864_0 ctx) (generatedRoot.w_865_0 ctx) (generatedRoot.w_866_0 ctx) (generatedRoot.w_867_0 ctx) (generatedRoot.w_868_0 ctx)) :
    generatedRoot.Facts_34 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_35  (ctx : generatedRootCtx) : Prop where
  t1120 : 0 ≤ 75
  t1121 : 75 < 100
  t1122 : 100 ≤ 100
  h869_0 : MxxRuntime.sliceMatrix (generatedRoot.w_862_0 ctx) 0 2 75 100 (generatedRoot.w_869_0 ctx)
  h870_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_869_0 ctx) (generatedRoot.w_870_0 ctx)
  h871_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_759_0 ctx), (generatedRoot.w_763_0 ctx), (generatedRoot.w_767_0 ctx), (generatedRoot.w_771_0 ctx), (generatedRoot.w_775_0 ctx), (generatedRoot.w_779_0 ctx), (generatedRoot.w_784_0 ctx), (generatedRoot.w_789_0 ctx), (generatedRoot.w_794_0 ctx), (generatedRoot.w_799_0 ctx), (generatedRoot.w_804_0 ctx), (generatedRoot.w_809_0 ctx), (generatedRoot.w_814_0 ctx), (generatedRoot.w_819_0 ctx), (generatedRoot.w_824_0 ctx), (generatedRoot.w_829_0 ctx), (generatedRoot.w_834_0 ctx), (generatedRoot.w_839_0 ctx), (generatedRoot.w_844_0 ctx), (generatedRoot.w_849_0 ctx), (generatedRoot.w_854_0 ctx), (generatedRoot.w_859_0 ctx), (generatedRoot.w_864_0 ctx), (generatedRoot.w_866_0 ctx), (generatedRoot.w_868_0 ctx), (generatedRoot.w_870_0 ctx)] (generatedRoot.w_871_0 ctx)
  t1126 : (36) = 36
  h872_3 : (∀ i : Fin 36, parallel_generatedRoot_872 backend hashModel params i ((generatedRoot.w_635_0 ctx), ((generatedRoot.w_660_0 ctx), ((generatedRoot.w_660_1 ctx), ((generatedRoot.w_707_0 ctx), ((generatedRoot.w_660_2 ctx), ((generatedRoot.w_660_3 ctx), ((generatedRoot.w_754_0 ctx), ((generatedRoot.w_660_4 ctx), ((generatedRoot.w_660_5 ctx), ((generatedRoot.w_871_0 ctx), ((generatedRoot.w_755_0 ctx), ()))))))))))) (((generatedRoot.w_872_0 ctx) i), (((generatedRoot.w_872_1 ctx) i), (((generatedRoot.w_872_2 ctx) i), (((generatedRoot.w_872_3 ctx) i), ())))))
  t1128 : 0 ≤ (0)
  t1129 : (0) < 900
  h882_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_882_0 ctx)
  h883_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_883_0 ctx)
  h886_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_886_0 ctx)
  h887_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_887_0 ctx)
  t1134 : (14400) = 14400
  h891_5 : (∀ i : Fin 14400, ∃ w_879_0_gathered : mxxMat0, ∃ w_880_0_gathered : mxxMat0, ∃ w_885_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_881_0 ctx) i) ∧ ((generatedRoot.w_881_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_879_0 ctx) ((generatedRoot.w_881_0 ctx) i) w_879_0_gathered ∧ 0 ≤ ((generatedRoot.w_881_0 ctx) i) ∧ ((generatedRoot.w_881_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_880_0 ctx) ((generatedRoot.w_881_0 ctx) i) w_880_0_gathered ∧ 0 ≤ ((generatedRoot.w_881_0 ctx) i) ∧ ((generatedRoot.w_881_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_885_0 ctx) ((generatedRoot.w_881_0 ctx) i) w_885_0_gathered ∧ parallel_generatedRoot_891 backend hashModel params i (((generatedRoot.w_881_0 ctx) i), (w_879_0_gathered, ((generatedRoot.w_7_0 ctx), (w_880_0_gathered, ((generatedRoot.w_882_0 ctx), ((generatedRoot.w_883_0 ctx), ((generatedRoot.w_884_0 ctx), (w_885_0_gathered, ((generatedRoot.w_886_0 ctx), ((generatedRoot.w_887_0 ctx), ((generatedRoot.w_888_0 ctx), ((generatedRoot.w_26_0 ctx), (((generatedRoot.w_889_0 ctx) i), (((generatedRoot.w_890_0 ctx) i), ())))))))))))))) (((generatedRoot.w_891_0 ctx) i), (((generatedRoot.w_891_1 ctx) i), (((generatedRoot.w_891_2 ctx) i), (((generatedRoot.w_891_3 ctx) i), (((generatedRoot.w_891_4 ctx) i), (((generatedRoot.w_891_5 ctx) i), ())))))))
  t1136 : (7200) = 7200
  h892_5 : (∀ i : Fin 7200, parallel_generatedRoot_892 backend hashModel params i ((generatedRoot.w_891_0 ctx), ((generatedRoot.w_891_1 ctx), ((generatedRoot.w_891_2 ctx), ((generatedRoot.w_891_3 ctx), ((generatedRoot.w_891_4 ctx), ((generatedRoot.w_891_5 ctx), ())))))) (((generatedRoot.w_892_0 ctx) i), (((generatedRoot.w_892_1 ctx) i), (((generatedRoot.w_892_2 ctx) i), (((generatedRoot.w_892_3 ctx) i), (((generatedRoot.w_892_4 ctx) i), (((generatedRoot.w_892_5 ctx) i), ())))))))
  t1138 : (14400) = 14400
  h895_5 : (∀ i : Fin 14400, ∃ w_879_0_gathered : mxxMat0, ∃ w_880_0_gathered : mxxMat0, ∃ w_885_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_878_0 ctx) i) ∧ ((generatedRoot.w_878_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_879_0 ctx) ((generatedRoot.w_878_0 ctx) i) w_879_0_gathered ∧ 0 ≤ ((generatedRoot.w_878_0 ctx) i) ∧ ((generatedRoot.w_878_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_880_0 ctx) ((generatedRoot.w_878_0 ctx) i) w_880_0_gathered ∧ 0 ≤ ((generatedRoot.w_878_0 ctx) i) ∧ ((generatedRoot.w_878_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_885_0 ctx) ((generatedRoot.w_878_0 ctx) i) w_885_0_gathered ∧ parallel_generatedRoot_895 backend hashModel params i (((generatedRoot.w_878_0 ctx) i), (w_879_0_gathered, ((generatedRoot.w_7_0 ctx), (w_880_0_gathered, ((generatedRoot.w_892_0 ctx), ((generatedRoot.w_892_2 ctx), ((generatedRoot.w_892_4 ctx), (w_885_0_gathered, ((generatedRoot.w_892_1 ctx), ((generatedRoot.w_892_3 ctx), ((generatedRoot.w_892_5 ctx), ((generatedRoot.w_26_0 ctx), (((generatedRoot.w_893_0 ctx) i), (((generatedRoot.w_894_0 ctx) i), ())))))))))))))) (((generatedRoot.w_895_0 ctx) i), (((generatedRoot.w_895_1 ctx) i), (((generatedRoot.w_895_2 ctx) i), (((generatedRoot.w_895_3 ctx) i), (((generatedRoot.w_895_4 ctx) i), (((generatedRoot.w_895_5 ctx) i), ())))))))
  t1140 : (7200) = 7200
  h896_5 : (∀ i : Fin 7200, parallel_generatedRoot_896 backend hashModel params i ((generatedRoot.w_895_0 ctx), ((generatedRoot.w_895_1 ctx), ((generatedRoot.w_895_2 ctx), ((generatedRoot.w_895_3 ctx), ((generatedRoot.w_895_4 ctx), ((generatedRoot.w_895_5 ctx), ())))))) (((generatedRoot.w_896_0 ctx) i), (((generatedRoot.w_896_1 ctx) i), (((generatedRoot.w_896_2 ctx) i), (((generatedRoot.w_896_3 ctx) i), (((generatedRoot.w_896_4 ctx) i), (((generatedRoot.w_896_5 ctx) i), ())))))))
  t1142 : (7200) = 7200
  h897_5 : (∀ i : Fin 7200, parallel_generatedRoot_897 backend hashModel params i ((generatedRoot.w_874_0 ctx), ((generatedRoot.w_875_0 ctx), ((generatedRoot.w_876_0 ctx), ((generatedRoot.w_877_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_14_0 ctx), (((generatedRoot.w_896_4 ctx) i), (((generatedRoot.w_896_0 ctx) i), (((generatedRoot.w_896_2 ctx) i), (((generatedRoot.w_896_5 ctx) i), (((generatedRoot.w_896_1 ctx) i), (((generatedRoot.w_896_3 ctx) i), ()))))))))))))) (((generatedRoot.w_897_0 ctx) i), (((generatedRoot.w_897_1 ctx) i), (((generatedRoot.w_897_2 ctx) i), (((generatedRoot.w_897_3 ctx) i), (((generatedRoot.w_897_4 ctx) i), (((generatedRoot.w_897_5 ctx) i), ())))))))
  t1144 : (7200) = 7200
  h898_5 : (∀ i : Fin 7200, parallel_generatedRoot_898 backend hashModel params i ((generatedRoot.w_897_0 ctx), ((generatedRoot.w_897_1 ctx), ((generatedRoot.w_897_2 ctx), ((generatedRoot.w_897_3 ctx), ((generatedRoot.w_897_4 ctx), ((generatedRoot.w_897_5 ctx), ())))))) (((generatedRoot.w_898_0 ctx) i), (((generatedRoot.w_898_1 ctx) i), (((generatedRoot.w_898_2 ctx) i), (((generatedRoot.w_898_3 ctx) i), (((generatedRoot.w_898_4 ctx) i), (((generatedRoot.w_898_5 ctx) i), ())))))))
  h899_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_0 ctx) (0) (generatedRoot.w_899_0 ctx)
  h900_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_0 ctx) (1) (generatedRoot.w_900_0 ctx)
  h902_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_1 ctx) (0) (generatedRoot.w_902_0 ctx)
  h903_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_1 ctx) (1) (generatedRoot.w_903_0 ctx)
  h905_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_2 ctx) (0) (generatedRoot.w_905_0 ctx)
  h906_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_2 ctx) (1) (generatedRoot.w_906_0 ctx)

theorem generatedRoot.facts_35  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_35 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_755_0 ctx) (generatedRoot.w_874_0 ctx) (generatedRoot.w_875_0 ctx) (generatedRoot.w_876_0 ctx) (generatedRoot.w_877_0 ctx) (generatedRoot.w_878_0 ctx) (generatedRoot.w_879_0 ctx) (generatedRoot.w_880_0 ctx) (generatedRoot.w_881_0 ctx) (generatedRoot.w_884_0 ctx) (generatedRoot.w_885_0 ctx) (generatedRoot.w_888_0 ctx) (generatedRoot.w_889_0 ctx) (generatedRoot.w_890_0 ctx) (generatedRoot.w_893_0 ctx) (generatedRoot.w_894_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_660_0 ctx) (generatedRoot.w_660_1 ctx) (generatedRoot.w_660_2 ctx) (generatedRoot.w_660_3 ctx) (generatedRoot.w_660_4 ctx) (generatedRoot.w_660_5 ctx) (generatedRoot.w_707_0 ctx) (generatedRoot.w_754_0 ctx) (generatedRoot.w_759_0 ctx) (generatedRoot.w_763_0 ctx) (generatedRoot.w_767_0 ctx) (generatedRoot.w_771_0 ctx) (generatedRoot.w_775_0 ctx) (generatedRoot.w_779_0 ctx) (generatedRoot.w_784_0 ctx) (generatedRoot.w_789_0 ctx) (generatedRoot.w_794_0 ctx) (generatedRoot.w_799_0 ctx) (generatedRoot.w_804_0 ctx) (generatedRoot.w_809_0 ctx) (generatedRoot.w_814_0 ctx) (generatedRoot.w_819_0 ctx) (generatedRoot.w_824_0 ctx) (generatedRoot.w_829_0 ctx) (generatedRoot.w_834_0 ctx) (generatedRoot.w_839_0 ctx) (generatedRoot.w_844_0 ctx) (generatedRoot.w_849_0 ctx) (generatedRoot.w_854_0 ctx) (generatedRoot.w_859_0 ctx) (generatedRoot.w_862_0 ctx) (generatedRoot.w_864_0 ctx) (generatedRoot.w_866_0 ctx) (generatedRoot.w_868_0 ctx) (generatedRoot.w_869_0 ctx) (generatedRoot.w_870_0 ctx) (generatedRoot.w_871_0 ctx) (generatedRoot.w_872_0 ctx) (generatedRoot.w_872_1 ctx) (generatedRoot.w_872_2 ctx) (generatedRoot.w_872_3 ctx) (generatedRoot.w_882_0 ctx) (generatedRoot.w_883_0 ctx) (generatedRoot.w_886_0 ctx) (generatedRoot.w_887_0 ctx) (generatedRoot.w_891_0 ctx) (generatedRoot.w_891_1 ctx) (generatedRoot.w_891_2 ctx) (generatedRoot.w_891_3 ctx) (generatedRoot.w_891_4 ctx) (generatedRoot.w_891_5 ctx) (generatedRoot.w_892_0 ctx) (generatedRoot.w_892_1 ctx) (generatedRoot.w_892_2 ctx) (generatedRoot.w_892_3 ctx) (generatedRoot.w_892_4 ctx) (generatedRoot.w_892_5 ctx) (generatedRoot.w_895_0 ctx) (generatedRoot.w_895_1 ctx) (generatedRoot.w_895_2 ctx) (generatedRoot.w_895_3 ctx) (generatedRoot.w_895_4 ctx) (generatedRoot.w_895_5 ctx) (generatedRoot.w_896_0 ctx) (generatedRoot.w_896_1 ctx) (generatedRoot.w_896_2 ctx) (generatedRoot.w_896_3 ctx) (generatedRoot.w_896_4 ctx) (generatedRoot.w_896_5 ctx) (generatedRoot.w_897_0 ctx) (generatedRoot.w_897_1 ctx) (generatedRoot.w_897_2 ctx) (generatedRoot.w_897_3 ctx) (generatedRoot.w_897_4 ctx) (generatedRoot.w_897_5 ctx) (generatedRoot.w_898_0 ctx) (generatedRoot.w_898_1 ctx) (generatedRoot.w_898_2 ctx) (generatedRoot.w_898_3 ctx) (generatedRoot.w_898_4 ctx) (generatedRoot.w_898_5 ctx) (generatedRoot.w_899_0 ctx) (generatedRoot.w_900_0 ctx) (generatedRoot.w_902_0 ctx) (generatedRoot.w_903_0 ctx) (generatedRoot.w_905_0 ctx) (generatedRoot.w_906_0 ctx)) :
    generatedRoot.Facts_35 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_36  (ctx : generatedRootCtx) : Prop where
  h908_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_3 ctx) (0) (generatedRoot.w_908_0 ctx)
  h909_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_3 ctx) (1) (generatedRoot.w_909_0 ctx)
  h911_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_4 ctx) (0) (generatedRoot.w_911_0 ctx)
  h912_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_4 ctx) (1) (generatedRoot.w_912_0 ctx)
  h914_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_5 ctx) (0) (generatedRoot.w_914_0 ctx)
  h915_0 : MxxRuntime.familyGetStatic (generatedRoot.w_898_5 ctx) (1) (generatedRoot.w_915_0 ctx)
  t1158 : 0 ≤ (6)
  h917_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_917 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_898_0 ctx), ((generatedRoot.w_898_1 ctx), ((generatedRoot.w_898_2 ctx), ((generatedRoot.w_898_3 ctx), ((generatedRoot.w_898_4 ctx), ((generatedRoot.w_898_5 ctx), ())))))))))))) next) (Int.toNat (6)) ((generatedRoot.w_901_0 ctx), ((generatedRoot.w_904_0 ctx), ((generatedRoot.w_907_0 ctx), ((generatedRoot.w_910_0 ctx), ((generatedRoot.w_913_0 ctx), ((generatedRoot.w_916_0 ctx), ())))))) ((generatedRoot.w_917_0 ctx), ((generatedRoot.w_917_1 ctx), ((generatedRoot.w_917_2 ctx), ((generatedRoot.w_917_3 ctx), ((generatedRoot.w_917_4 ctx), ((generatedRoot.w_917_5 ctx), ()))))))
  h918_0 : MxxRuntime.tensorRuns (generatedRoot.w_873_0 ctx) (generatedRoot.w_917_0 ctx) (generatedRoot.w_918_0 ctx)
  t1161 : 0 ≤ (0)
  t1162 : (0) < 900
  h920_0 : MxxRuntime.tensorRuns (generatedRoot.w_919_0 ctx) (generatedRoot.w_917_1 ctx) (generatedRoot.w_920_0 ctx)
  t1164 : 0 ≤ (0)
  t1165 : (0) < 900
  h922_0 : MxxRuntime.tensorRuns (generatedRoot.w_921_0 ctx) (generatedRoot.w_917_2 ctx) (generatedRoot.w_922_0 ctx)
  t1167 : 0 ≤ (0)
  t1168 : (0) < 900
  h924_0 : MxxRuntime.tensorRuns (generatedRoot.w_923_0 ctx) (generatedRoot.w_917_3 ctx) (generatedRoot.w_924_0 ctx)
  t1170 : 0 ≤ (0)
  t1171 : (0) < 900
  h926_0 : MxxRuntime.tensorRuns (generatedRoot.w_925_0 ctx) (generatedRoot.w_917_4 ctx) (generatedRoot.w_926_0 ctx)
  t1173 : 0 ≤ (0)
  t1174 : (0) < 900
  h928_0 : MxxRuntime.tensorRuns (generatedRoot.w_927_0 ctx) (generatedRoot.w_917_5 ctx) (generatedRoot.w_928_0 ctx)
  t1176 : 0 ≤ (899)
  h929_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) => sequential_generatedRoot_929 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_898_0 ctx), ((generatedRoot.w_898_1 ctx), ((generatedRoot.w_898_2 ctx), ((generatedRoot.w_898_3 ctx), ((generatedRoot.w_898_4 ctx), ((generatedRoot.w_898_5 ctx), ())))))))))))) next) (Int.toNat (899)) ((generatedRoot.w_918_0 ctx), ((generatedRoot.w_920_0 ctx), ((generatedRoot.w_922_0 ctx), ((generatedRoot.w_924_0 ctx), ((generatedRoot.w_926_0 ctx), ((generatedRoot.w_928_0 ctx), ())))))) ((generatedRoot.w_929_0 ctx), ((generatedRoot.w_929_1 ctx), ((generatedRoot.w_929_2 ctx), ((generatedRoot.w_929_3 ctx), ((generatedRoot.w_929_4 ctx), ((generatedRoot.w_929_5 ctx), ()))))))
  t1178 : 0 ≤ (0)
  t1179 : (0) < 4500
  h937_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_937_0 ctx)
  h938_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_938_0 ctx)
  h940_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_940_0 ctx)
  h941_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_941_0 ctx)

theorem generatedRoot.facts_36  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_36 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_873_0 ctx) (generatedRoot.w_898_0 ctx) (generatedRoot.w_898_1 ctx) (generatedRoot.w_898_2 ctx) (generatedRoot.w_898_3 ctx) (generatedRoot.w_898_4 ctx) (generatedRoot.w_898_5 ctx) (generatedRoot.w_901_0 ctx) (generatedRoot.w_904_0 ctx) (generatedRoot.w_907_0 ctx) (generatedRoot.w_908_0 ctx) (generatedRoot.w_909_0 ctx) (generatedRoot.w_910_0 ctx) (generatedRoot.w_911_0 ctx) (generatedRoot.w_912_0 ctx) (generatedRoot.w_913_0 ctx) (generatedRoot.w_914_0 ctx) (generatedRoot.w_915_0 ctx) (generatedRoot.w_916_0 ctx) (generatedRoot.w_917_0 ctx) (generatedRoot.w_917_1 ctx) (generatedRoot.w_917_2 ctx) (generatedRoot.w_917_3 ctx) (generatedRoot.w_917_4 ctx) (generatedRoot.w_917_5 ctx) (generatedRoot.w_918_0 ctx) (generatedRoot.w_919_0 ctx) (generatedRoot.w_920_0 ctx) (generatedRoot.w_921_0 ctx) (generatedRoot.w_922_0 ctx) (generatedRoot.w_923_0 ctx) (generatedRoot.w_924_0 ctx) (generatedRoot.w_925_0 ctx) (generatedRoot.w_926_0 ctx) (generatedRoot.w_927_0 ctx) (generatedRoot.w_928_0 ctx) (generatedRoot.w_929_0 ctx) (generatedRoot.w_929_1 ctx) (generatedRoot.w_929_2 ctx) (generatedRoot.w_929_3 ctx) (generatedRoot.w_929_4 ctx) (generatedRoot.w_929_5 ctx) (generatedRoot.w_937_0 ctx) (generatedRoot.w_938_0 ctx) (generatedRoot.w_940_0 ctx) (generatedRoot.w_941_0 ctx)) :
    generatedRoot.Facts_36 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_37  (ctx : generatedRootCtx) : Prop where
  t1184 : (72000) = 72000
  h945_5 : (∀ i : Fin 72000, ∃ w_100_0_gathered : mxxMat0, ∃ w_105_0_gathered : mxxMat0, ∃ w_119_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_936_0 ctx) i) ∧ ((generatedRoot.w_936_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_100_0 ctx) ((generatedRoot.w_936_0 ctx) i) w_100_0_gathered ∧ 0 ≤ ((generatedRoot.w_936_0 ctx) i) ∧ ((generatedRoot.w_936_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_105_0 ctx) ((generatedRoot.w_936_0 ctx) i) w_105_0_gathered ∧ 0 ≤ ((generatedRoot.w_936_0 ctx) i) ∧ ((generatedRoot.w_936_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_119_0 ctx) ((generatedRoot.w_936_0 ctx) i) w_119_0_gathered ∧ parallel_generatedRoot_945 backend hashModel params i (((generatedRoot.w_936_0 ctx) i), (w_100_0_gathered, ((generatedRoot.w_7_0 ctx), (w_105_0_gathered, ((generatedRoot.w_937_0 ctx), ((generatedRoot.w_938_0 ctx), ((generatedRoot.w_939_0 ctx), (w_119_0_gathered, ((generatedRoot.w_940_0 ctx), ((generatedRoot.w_941_0 ctx), ((generatedRoot.w_942_0 ctx), ((generatedRoot.w_26_0 ctx), (((generatedRoot.w_943_0 ctx) i), (((generatedRoot.w_944_0 ctx) i), ())))))))))))))) (((generatedRoot.w_945_0 ctx) i), (((generatedRoot.w_945_1 ctx) i), (((generatedRoot.w_945_2 ctx) i), (((generatedRoot.w_945_3 ctx) i), (((generatedRoot.w_945_4 ctx) i), (((generatedRoot.w_945_5 ctx) i), ())))))))
  t1186 : (36000) = 36000
  h946_5 : (∀ i : Fin 36000, parallel_generatedRoot_946 backend hashModel params i ((generatedRoot.w_945_0 ctx), ((generatedRoot.w_945_1 ctx), ((generatedRoot.w_945_2 ctx), ((generatedRoot.w_945_3 ctx), ((generatedRoot.w_945_4 ctx), ((generatedRoot.w_945_5 ctx), ())))))) (((generatedRoot.w_946_0 ctx) i), (((generatedRoot.w_946_1 ctx) i), (((generatedRoot.w_946_2 ctx) i), (((generatedRoot.w_946_3 ctx) i), (((generatedRoot.w_946_4 ctx) i), (((generatedRoot.w_946_5 ctx) i), ())))))))
  t1188 : (72000) = 72000
  h949_5 : (∀ i : Fin 72000, ∃ w_100_0_gathered : mxxMat0, ∃ w_105_0_gathered : mxxMat0, ∃ w_119_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_935_0 ctx) i) ∧ ((generatedRoot.w_935_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_100_0 ctx) ((generatedRoot.w_935_0 ctx) i) w_100_0_gathered ∧ 0 ≤ ((generatedRoot.w_935_0 ctx) i) ∧ ((generatedRoot.w_935_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_105_0 ctx) ((generatedRoot.w_935_0 ctx) i) w_105_0_gathered ∧ 0 ≤ ((generatedRoot.w_935_0 ctx) i) ∧ ((generatedRoot.w_935_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_119_0 ctx) ((generatedRoot.w_935_0 ctx) i) w_119_0_gathered ∧ parallel_generatedRoot_949 backend hashModel params i (((generatedRoot.w_935_0 ctx) i), (w_100_0_gathered, ((generatedRoot.w_7_0 ctx), (w_105_0_gathered, ((generatedRoot.w_946_0 ctx), ((generatedRoot.w_946_2 ctx), ((generatedRoot.w_946_4 ctx), (w_119_0_gathered, ((generatedRoot.w_946_1 ctx), ((generatedRoot.w_946_3 ctx), ((generatedRoot.w_946_5 ctx), ((generatedRoot.w_26_0 ctx), (((generatedRoot.w_947_0 ctx) i), (((generatedRoot.w_948_0 ctx) i), ())))))))))))))) (((generatedRoot.w_949_0 ctx) i), (((generatedRoot.w_949_1 ctx) i), (((generatedRoot.w_949_2 ctx) i), (((generatedRoot.w_949_3 ctx) i), (((generatedRoot.w_949_4 ctx) i), (((generatedRoot.w_949_5 ctx) i), ())))))))
  t1190 : (36000) = 36000
  h950_5 : (∀ i : Fin 36000, parallel_generatedRoot_950 backend hashModel params i ((generatedRoot.w_949_0 ctx), ((generatedRoot.w_949_1 ctx), ((generatedRoot.w_949_2 ctx), ((generatedRoot.w_949_3 ctx), ((generatedRoot.w_949_4 ctx), ((generatedRoot.w_949_5 ctx), ())))))) (((generatedRoot.w_950_0 ctx) i), (((generatedRoot.w_950_1 ctx) i), (((generatedRoot.w_950_2 ctx) i), (((generatedRoot.w_950_3 ctx) i), (((generatedRoot.w_950_4 ctx) i), (((generatedRoot.w_950_5 ctx) i), ())))))))
  t1192 : (36000) = 36000
  h951_5 : (∀ i : Fin 36000, parallel_generatedRoot_951 backend hashModel params i ((generatedRoot.w_931_0 ctx), ((generatedRoot.w_932_0 ctx), ((generatedRoot.w_933_0 ctx), ((generatedRoot.w_934_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_14_0 ctx), (((generatedRoot.w_950_4 ctx) i), (((generatedRoot.w_950_0 ctx) i), (((generatedRoot.w_950_2 ctx) i), (((generatedRoot.w_950_5 ctx) i), (((generatedRoot.w_950_1 ctx) i), (((generatedRoot.w_950_3 ctx) i), ()))))))))))))) (((generatedRoot.w_951_0 ctx) i), (((generatedRoot.w_951_1 ctx) i), (((generatedRoot.w_951_2 ctx) i), (((generatedRoot.w_951_3 ctx) i), (((generatedRoot.w_951_4 ctx) i), (((generatedRoot.w_951_5 ctx) i), ())))))))
  t1194 : (36000) = 36000
  h952_5 : (∀ i : Fin 36000, parallel_generatedRoot_952 backend hashModel params i ((generatedRoot.w_951_0 ctx), ((generatedRoot.w_951_1 ctx), ((generatedRoot.w_951_2 ctx), ((generatedRoot.w_951_3 ctx), ((generatedRoot.w_951_4 ctx), ((generatedRoot.w_951_5 ctx), ())))))) (((generatedRoot.w_952_0 ctx) i), (((generatedRoot.w_952_1 ctx) i), (((generatedRoot.w_952_2 ctx) i), (((generatedRoot.w_952_3 ctx) i), (((generatedRoot.w_952_4 ctx) i), (((generatedRoot.w_952_5 ctx) i), ())))))))
  h953_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_0 ctx) (0) (generatedRoot.w_953_0 ctx)
  h954_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_0 ctx) (1) (generatedRoot.w_954_0 ctx)
  h956_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_1 ctx) (0) (generatedRoot.w_956_0 ctx)
  h957_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_1 ctx) (1) (generatedRoot.w_957_0 ctx)
  h959_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_2 ctx) (0) (generatedRoot.w_959_0 ctx)
  h960_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_2 ctx) (1) (generatedRoot.w_960_0 ctx)
  h962_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_3 ctx) (0) (generatedRoot.w_962_0 ctx)
  h963_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_3 ctx) (1) (generatedRoot.w_963_0 ctx)
  h965_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_4 ctx) (0) (generatedRoot.w_965_0 ctx)
  h966_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_4 ctx) (1) (generatedRoot.w_966_0 ctx)
  h968_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_5 ctx) (0) (generatedRoot.w_968_0 ctx)
  h969_0 : MxxRuntime.familyGetStatic (generatedRoot.w_952_5 ctx) (1) (generatedRoot.w_969_0 ctx)
  t1208 : 0 ≤ (6)
  h971_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_971 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_952_0 ctx), ((generatedRoot.w_952_1 ctx), ((generatedRoot.w_952_2 ctx), ((generatedRoot.w_952_3 ctx), ((generatedRoot.w_952_4 ctx), ((generatedRoot.w_952_5 ctx), ())))))))))))) next) (Int.toNat (6)) ((generatedRoot.w_955_0 ctx), ((generatedRoot.w_958_0 ctx), ((generatedRoot.w_961_0 ctx), ((generatedRoot.w_964_0 ctx), ((generatedRoot.w_967_0 ctx), ((generatedRoot.w_970_0 ctx), ())))))) ((generatedRoot.w_971_0 ctx), ((generatedRoot.w_971_1 ctx), ((generatedRoot.w_971_2 ctx), ((generatedRoot.w_971_3 ctx), ((generatedRoot.w_971_4 ctx), ((generatedRoot.w_971_5 ctx), ()))))))
  h972_0 : MxxRuntime.tensorRuns (generatedRoot.w_930_0 ctx) (generatedRoot.w_971_0 ctx) (generatedRoot.w_972_0 ctx)
  t1211 : 0 ≤ (0)
  t1212 : (0) < 4500
  h974_0 : MxxRuntime.tensorRuns (generatedRoot.w_973_0 ctx) (generatedRoot.w_971_1 ctx) (generatedRoot.w_974_0 ctx)
  t1214 : 0 ≤ (0)
  t1215 : (0) < 4500

theorem generatedRoot.facts_37  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_37 backend hashModel params (generatedRoot.w_7_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_931_0 ctx) (generatedRoot.w_932_0 ctx) (generatedRoot.w_933_0 ctx) (generatedRoot.w_934_0 ctx) (generatedRoot.w_935_0 ctx) (generatedRoot.w_936_0 ctx) (generatedRoot.w_939_0 ctx) (generatedRoot.w_942_0 ctx) (generatedRoot.w_943_0 ctx) (generatedRoot.w_944_0 ctx) (generatedRoot.w_947_0 ctx) (generatedRoot.w_948_0 ctx) (generatedRoot.w_930_0 ctx) (generatedRoot.w_937_0 ctx) (generatedRoot.w_938_0 ctx) (generatedRoot.w_940_0 ctx) (generatedRoot.w_941_0 ctx) (generatedRoot.w_945_0 ctx) (generatedRoot.w_945_1 ctx) (generatedRoot.w_945_2 ctx) (generatedRoot.w_945_3 ctx) (generatedRoot.w_945_4 ctx) (generatedRoot.w_945_5 ctx) (generatedRoot.w_946_0 ctx) (generatedRoot.w_946_1 ctx) (generatedRoot.w_946_2 ctx) (generatedRoot.w_946_3 ctx) (generatedRoot.w_946_4 ctx) (generatedRoot.w_946_5 ctx) (generatedRoot.w_949_0 ctx) (generatedRoot.w_949_1 ctx) (generatedRoot.w_949_2 ctx) (generatedRoot.w_949_3 ctx) (generatedRoot.w_949_4 ctx) (generatedRoot.w_949_5 ctx) (generatedRoot.w_950_0 ctx) (generatedRoot.w_950_1 ctx) (generatedRoot.w_950_2 ctx) (generatedRoot.w_950_3 ctx) (generatedRoot.w_950_4 ctx) (generatedRoot.w_950_5 ctx) (generatedRoot.w_951_0 ctx) (generatedRoot.w_951_1 ctx) (generatedRoot.w_951_2 ctx) (generatedRoot.w_951_3 ctx) (generatedRoot.w_951_4 ctx) (generatedRoot.w_951_5 ctx) (generatedRoot.w_952_0 ctx) (generatedRoot.w_952_1 ctx) (generatedRoot.w_952_2 ctx) (generatedRoot.w_952_3 ctx) (generatedRoot.w_952_4 ctx) (generatedRoot.w_952_5 ctx) (generatedRoot.w_953_0 ctx) (generatedRoot.w_954_0 ctx) (generatedRoot.w_955_0 ctx) (generatedRoot.w_956_0 ctx) (generatedRoot.w_957_0 ctx) (generatedRoot.w_958_0 ctx) (generatedRoot.w_959_0 ctx) (generatedRoot.w_960_0 ctx) (generatedRoot.w_961_0 ctx) (generatedRoot.w_962_0 ctx) (generatedRoot.w_963_0 ctx) (generatedRoot.w_964_0 ctx) (generatedRoot.w_965_0 ctx) (generatedRoot.w_966_0 ctx) (generatedRoot.w_967_0 ctx) (generatedRoot.w_968_0 ctx) (generatedRoot.w_969_0 ctx) (generatedRoot.w_970_0 ctx) (generatedRoot.w_971_0 ctx) (generatedRoot.w_971_1 ctx) (generatedRoot.w_971_2 ctx) (generatedRoot.w_971_3 ctx) (generatedRoot.w_971_4 ctx) (generatedRoot.w_971_5 ctx) (generatedRoot.w_972_0 ctx) (generatedRoot.w_973_0 ctx) (generatedRoot.w_974_0 ctx)) :
    generatedRoot.Facts_37 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_38  (ctx : generatedRootCtx) : Prop where
  h976_0 : MxxRuntime.tensorRuns (generatedRoot.w_975_0 ctx) (generatedRoot.w_971_2 ctx) (generatedRoot.w_976_0 ctx)
  t1217 : 0 ≤ (0)
  t1218 : (0) < 4500
  h978_0 : MxxRuntime.tensorRuns (generatedRoot.w_977_0 ctx) (generatedRoot.w_971_3 ctx) (generatedRoot.w_978_0 ctx)
  t1220 : 0 ≤ (0)
  t1221 : (0) < 4500
  h980_0 : MxxRuntime.tensorRuns (generatedRoot.w_979_0 ctx) (generatedRoot.w_971_4 ctx) (generatedRoot.w_980_0 ctx)
  t1223 : 0 ≤ (0)
  t1224 : (0) < 4500
  h982_0 : MxxRuntime.tensorRuns (generatedRoot.w_981_0 ctx) (generatedRoot.w_971_5 ctx) (generatedRoot.w_982_0 ctx)
  t1226 : 0 ≤ (4499)
  h983_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) => sequential_generatedRoot_983 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_952_0 ctx), ((generatedRoot.w_952_1 ctx), ((generatedRoot.w_952_2 ctx), ((generatedRoot.w_952_3 ctx), ((generatedRoot.w_952_4 ctx), ((generatedRoot.w_952_5 ctx), ())))))))))))) next) (Int.toNat (4499)) ((generatedRoot.w_972_0 ctx), ((generatedRoot.w_974_0 ctx), ((generatedRoot.w_976_0 ctx), ((generatedRoot.w_978_0 ctx), ((generatedRoot.w_980_0 ctx), ((generatedRoot.w_982_0 ctx), ())))))) ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_983_1 ctx), ((generatedRoot.w_983_2 ctx), ((generatedRoot.w_983_3 ctx), ((generatedRoot.w_983_4 ctx), ((generatedRoot.w_983_5 ctx), ()))))))
  t1228 : (1) = 1
  h985_0 : (∀ i : Fin 1, parallel_generatedRoot_985 backend hashModel params i (generatedRoot.w_984_0 ctx) ((generatedRoot.w_985_0 ctx) i))
  t1230 : (36) = 36
  h991_0 : (∀ i : Fin 36, parallel_generatedRoot_991 backend hashModel params i ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_1_0 ctx) i), ((generatedRoot.w_63_0 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_872_3 ctx) i), (((generatedRoot.w_872_0 ctx) i), ((generatedRoot.w_929_5 ctx), ((generatedRoot.w_57_0 ctx), ((generatedRoot.w_983_5 ctx), ((generatedRoot.w_985_0 ctx), ((generatedRoot.w_983_1 ctx), ((generatedRoot.w_983_3 ctx), ((generatedRoot.w_986_0 ctx), (((generatedRoot.w_987_0 ctx) i), (((generatedRoot.w_988_0 ctx) i), (((generatedRoot.w_989_0 ctx) i), (((generatedRoot.w_990_0 ctx) i), ()))))))))))))))))) ((generatedRoot.w_991_0 ctx) i))
  h992_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (32) (generatedRoot.w_992_0 ctx)
  h993_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (33) (generatedRoot.w_993_0 ctx)
  h994_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (34) (generatedRoot.w_994_0 ctx)
  h995_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (35) (generatedRoot.w_995_0 ctx)
  h993_0_1 : MxxRuntime.concatColumns (generatedRoot.w_992_0 ctx) (generatedRoot.w_993_0 ctx) (generatedRoot.w_996_concat_1 ctx)
  h994_0_1 : MxxRuntime.concatColumns (generatedRoot.w_996_concat_1 ctx) (generatedRoot.w_994_0 ctx) (generatedRoot.w_996_concat_2 ctx)
  h996_0 : MxxRuntime.concatColumns (generatedRoot.w_996_concat_2 ctx) (generatedRoot.w_995_0 ctx) (generatedRoot.w_996_0 ctx)
  t1239 : 0 ≤ 0
  t1240 : 0 < 1
  t1241 : 1 ≤ 1
  t1242 : 0 ≤ 0
  t1243 : 0 < 54
  t1244 : 54 ≤ 100
  h997_0 : MxxRuntime.sliceMatrix (generatedRoot.w_996_0 ctx) 0 1 0 54 (generatedRoot.w_997_0 ctx)
  h998_0 : MxxRuntime.concatRows (generatedRoot.w_529_6 ctx) (generatedRoot.w_529_8 ctx) (generatedRoot.w_998_0 ctx)
  h999_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_998_0 ctx) (generatedRoot.w_999_0 ctx)

theorem generatedRoot.facts_38  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_38 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_987_0 ctx) (generatedRoot.w_988_0 ctx) (generatedRoot.w_989_0 ctx) (generatedRoot.w_990_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_529_6 ctx) (generatedRoot.w_529_8 ctx) (generatedRoot.w_872_0 ctx) (generatedRoot.w_872_3 ctx) (generatedRoot.w_929_5 ctx) (generatedRoot.w_952_0 ctx) (generatedRoot.w_952_1 ctx) (generatedRoot.w_952_2 ctx) (generatedRoot.w_952_3 ctx) (generatedRoot.w_952_4 ctx) (generatedRoot.w_952_5 ctx) (generatedRoot.w_971_2 ctx) (generatedRoot.w_971_3 ctx) (generatedRoot.w_971_4 ctx) (generatedRoot.w_971_5 ctx) (generatedRoot.w_972_0 ctx) (generatedRoot.w_974_0 ctx) (generatedRoot.w_975_0 ctx) (generatedRoot.w_976_0 ctx) (generatedRoot.w_977_0 ctx) (generatedRoot.w_978_0 ctx) (generatedRoot.w_979_0 ctx) (generatedRoot.w_980_0 ctx) (generatedRoot.w_981_0 ctx) (generatedRoot.w_982_0 ctx) (generatedRoot.w_983_0 ctx) (generatedRoot.w_983_1 ctx) (generatedRoot.w_983_2 ctx) (generatedRoot.w_983_3 ctx) (generatedRoot.w_983_4 ctx) (generatedRoot.w_983_5 ctx) (generatedRoot.w_984_0 ctx) (generatedRoot.w_985_0 ctx) (generatedRoot.w_986_0 ctx) (generatedRoot.w_991_0 ctx) (generatedRoot.w_992_0 ctx) (generatedRoot.w_993_0 ctx) (generatedRoot.w_994_0 ctx) (generatedRoot.w_995_0 ctx) (generatedRoot.w_996_0 ctx) (generatedRoot.w_996_concat_1 ctx) (generatedRoot.w_996_concat_2 ctx) (generatedRoot.w_997_0 ctx) (generatedRoot.w_998_0 ctx) (generatedRoot.w_999_0 ctx)) :
    generatedRoot.Facts_38 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_39  (ctx : generatedRootCtx) : Prop where
  h1000_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_999_0 ctx) (generatedRoot.w_1000_0 ctx)
  t1249 : 0 ≤ 1
  t1250 : 1 < 2
  t1251 : 2 ≤ 3
  t1252 : 0 ≤ 0
  t1253 : 0 < 25
  t1254 : 25 ≤ 25
  h1002_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1001_0 ctx) 1 2 0 25 (generatedRoot.w_1002_0 ctx)
  t1256 : 0 ≤ 2
  t1257 : 2 < 3
  t1258 : 3 ≤ 3
  t1259 : 0 ≤ 0
  t1260 : 0 < 25
  t1261 : 25 ≤ 25
  h1003_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1001_0 ctx) 2 3 0 25 (generatedRoot.w_1003_0 ctx)
  h1004_0 : MxxRuntime.concatRows (generatedRoot.w_1002_0 ctx) (generatedRoot.w_1003_0 ctx) (generatedRoot.w_1004_0 ctx)
  h1005_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1004_0 ctx) (generatedRoot.w_1005_0 ctx)
  h1006_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1005_0 ctx) (generatedRoot.w_1006_0 ctx)
  t1266 : 0 ≤ 1
  t1267 : 1 < 2
  t1268 : 2 ≤ 3
  t1269 : 0 ≤ 0
  t1270 : 0 < 25
  t1271 : 25 ≤ 25
  h1008_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1007_0 ctx) 1 2 0 25 (generatedRoot.w_1008_0 ctx)
  t1273 : 0 ≤ 2
  t1274 : 2 < 3
  t1275 : 3 ≤ 3
  t1276 : 0 ≤ 0
  t1277 : 0 < 25
  t1278 : 25 ≤ 25
  h1010_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1007_0 ctx) 2 3 0 25 (generatedRoot.w_1010_0 ctx)

theorem generatedRoot.facts_39  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_39 backend (generatedRoot.w_999_0 ctx) (generatedRoot.w_1000_0 ctx) (generatedRoot.w_1001_0 ctx) (generatedRoot.w_1002_0 ctx) (generatedRoot.w_1003_0 ctx) (generatedRoot.w_1004_0 ctx) (generatedRoot.w_1005_0 ctx) (generatedRoot.w_1006_0 ctx) (generatedRoot.w_1007_0 ctx) (generatedRoot.w_1008_0 ctx) (generatedRoot.w_1010_0 ctx)) :
    generatedRoot.Facts_39 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_40  (ctx : generatedRootCtx) : Prop where
  t1280 : 0 ≤ 0
  t1281 : 0 < 1
  t1282 : 1 ≤ 3
  t1283 : 0 ≤ 0
  t1284 : 0 < 25
  t1285 : 25 ≤ 25
  h1012_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1007_0 ctx) 0 1 0 25 (generatedRoot.w_1012_0 ctx)
  t1287 : 0 ≤ 0
  t1288 : 0 < 1
  t1289 : 1 ≤ 3
  t1290 : 0 ≤ 0
  t1291 : 0 < 25
  t1292 : 25 ≤ 25
  h1013_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1001_0 ctx) 0 1 0 25 (generatedRoot.w_1013_0 ctx)
  h1016_0 : MxxRuntime.concatRows (generatedRoot.w_587_6 ctx) (generatedRoot.w_587_8 ctx) (generatedRoot.w_1016_0 ctx)
  h1017_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1016_0 ctx) (generatedRoot.w_1017_0 ctx)
  h1018_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1017_0 ctx) (generatedRoot.w_1018_0 ctx)
  t1297 : 0 ≤ 1
  t1298 : 1 < 2
  t1299 : 2 ≤ 3
  t1300 : 0 ≤ 0
  t1301 : 0 < 25
  t1302 : 25 ≤ 25
  h1020_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1019_0 ctx) 1 2 0 25 (generatedRoot.w_1020_0 ctx)
  t1304 : 0 ≤ 2
  t1305 : 2 < 3
  t1306 : 3 ≤ 3
  t1307 : 0 ≤ 0
  t1308 : 0 < 25
  t1309 : 25 ≤ 25
  h1021_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1019_0 ctx) 2 3 0 25 (generatedRoot.w_1021_0 ctx)
  h1022_0 : MxxRuntime.concatRows (generatedRoot.w_1020_0 ctx) (generatedRoot.w_1021_0 ctx) (generatedRoot.w_1022_0 ctx)

theorem generatedRoot.facts_40  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_40 backend (generatedRoot.w_587_6 ctx) (generatedRoot.w_587_8 ctx) (generatedRoot.w_1001_0 ctx) (generatedRoot.w_1007_0 ctx) (generatedRoot.w_1012_0 ctx) (generatedRoot.w_1013_0 ctx) (generatedRoot.w_1016_0 ctx) (generatedRoot.w_1017_0 ctx) (generatedRoot.w_1018_0 ctx) (generatedRoot.w_1019_0 ctx) (generatedRoot.w_1020_0 ctx) (generatedRoot.w_1021_0 ctx) (generatedRoot.w_1022_0 ctx)) :
    generatedRoot.Facts_40 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_41  (ctx : generatedRootCtx) : Prop where
  h1023_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1022_0 ctx) (generatedRoot.w_1023_0 ctx)
  h1024_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1023_0 ctx) (generatedRoot.w_1024_0 ctx)
  t1314 : 0 ≤ 1
  t1315 : 1 < 2
  t1316 : 2 ≤ 3
  t1317 : 0 ≤ 0
  t1318 : 0 < 25
  t1319 : 25 ≤ 25
  h1026_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1025_0 ctx) 1 2 0 25 (generatedRoot.w_1026_0 ctx)
  t1321 : 0 ≤ 2
  t1322 : 2 < 3
  t1323 : 3 ≤ 3
  t1324 : 0 ≤ 0
  t1325 : 0 < 25
  t1326 : 25 ≤ 25
  h1028_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1025_0 ctx) 2 3 0 25 (generatedRoot.w_1028_0 ctx)
  t1328 : 0 ≤ 0
  t1329 : 0 < 1
  t1330 : 1 ≤ 3
  t1331 : 0 ≤ 0
  t1332 : 0 < 25
  t1333 : 25 ≤ 25
  h1030_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1025_0 ctx) 0 1 0 25 (generatedRoot.w_1030_0 ctx)
  t1335 : 0 ≤ 0
  t1336 : 0 < 1
  t1337 : 1 ≤ 3
  t1338 : 0 ≤ 0
  t1339 : 0 < 25
  t1340 : 25 ≤ 25
  h1031_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1019_0 ctx) 0 1 0 25 (generatedRoot.w_1031_0 ctx)
  h1034_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_1034_0 ctx)
  h1035_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_7_0 ctx)) (generatedRoot.w_1035_0 ctx)

theorem generatedRoot.facts_41  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_41 backend hashModel (generatedRoot.w_7_0 ctx) (generatedRoot.w_1019_0 ctx) (generatedRoot.w_1022_0 ctx) (generatedRoot.w_1023_0 ctx) (generatedRoot.w_1024_0 ctx) (generatedRoot.w_1025_0 ctx) (generatedRoot.w_1026_0 ctx) (generatedRoot.w_1028_0 ctx) (generatedRoot.w_1030_0 ctx) (generatedRoot.w_1031_0 ctx) (generatedRoot.w_1034_0 ctx) (generatedRoot.w_1035_0 ctx)) :
    generatedRoot.Facts_41 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_42  (ctx : generatedRootCtx) : Prop where
  h1036_0 : MxxRuntime.familyGetStatic (generatedRoot.w_16_0 ctx) (1) (generatedRoot.w_1036_0 ctx)
  h1037_0 : MxxRuntime.familyGetStatic (generatedRoot.w_22_0 ctx) (1) (generatedRoot.w_1037_0 ctx)
  h1038_0 : MxxRuntime.concatRows (generatedRoot.w_239_6 ctx) (generatedRoot.w_239_8 ctx) (generatedRoot.w_1038_0 ctx)
  h1039_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1038_0 ctx) (generatedRoot.w_1039_0 ctx)
  h1040_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1039_0 ctx) (generatedRoot.w_1040_0 ctx)
  t1349 : 0 ≤ 1
  t1350 : 1 < 2
  t1351 : 2 ≤ 3
  t1352 : 0 ≤ 0
  t1353 : 0 < 25
  t1354 : 25 ≤ 25
  h1042_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1041_0 ctx) 1 2 0 25 (generatedRoot.w_1042_0 ctx)
  t1356 : 0 ≤ 2
  t1357 : 2 < 3
  t1358 : 3 ≤ 3
  t1359 : 0 ≤ 0
  t1360 : 0 < 25
  t1361 : 25 ≤ 25
  h1043_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1041_0 ctx) 2 3 0 25 (generatedRoot.w_1043_0 ctx)
  h1044_0 : MxxRuntime.concatRows (generatedRoot.w_1042_0 ctx) (generatedRoot.w_1043_0 ctx) (generatedRoot.w_1044_0 ctx)
  h1045_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1044_0 ctx) (generatedRoot.w_1045_0 ctx)
  h1046_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1045_0 ctx) (generatedRoot.w_1046_0 ctx)
  t1366 : 0 ≤ 1
  t1367 : 1 < 2
  t1368 : 2 ≤ 3
  t1369 : 0 ≤ 0
  t1370 : 0 < 25
  t1371 : 25 ≤ 25
  h1048_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1047_0 ctx) 1 2 0 25 (generatedRoot.w_1048_0 ctx)
  t1373 : 0 ≤ 2
  t1374 : 2 < 3
  t1375 : 3 ≤ 3

theorem generatedRoot.facts_42  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_42 backend (generatedRoot.w_16_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_239_6 ctx) (generatedRoot.w_239_8 ctx) (generatedRoot.w_1036_0 ctx) (generatedRoot.w_1037_0 ctx) (generatedRoot.w_1038_0 ctx) (generatedRoot.w_1039_0 ctx) (generatedRoot.w_1040_0 ctx) (generatedRoot.w_1041_0 ctx) (generatedRoot.w_1042_0 ctx) (generatedRoot.w_1043_0 ctx) (generatedRoot.w_1044_0 ctx) (generatedRoot.w_1045_0 ctx) (generatedRoot.w_1046_0 ctx) (generatedRoot.w_1047_0 ctx) (generatedRoot.w_1048_0 ctx)) :
    generatedRoot.Facts_42 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_43  (ctx : generatedRootCtx) : Prop where
  t1376 : 0 ≤ 0
  t1377 : 0 < 25
  t1378 : 25 ≤ 25
  h1050_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1047_0 ctx) 2 3 0 25 (generatedRoot.w_1050_0 ctx)
  t1380 : 0 ≤ 0
  t1381 : 0 < 1
  t1382 : 1 ≤ 3
  t1383 : 0 ≤ 0
  t1384 : 0 < 25
  t1385 : 25 ≤ 25
  h1052_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1047_0 ctx) 0 1 0 25 (generatedRoot.w_1052_0 ctx)
  t1387 : 0 ≤ 0
  t1388 : 0 < 1
  t1389 : 1 ≤ 3
  t1390 : 0 ≤ 0
  t1391 : 0 < 25
  t1392 : 25 ≤ 25
  h1053_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1041_0 ctx) 0 1 0 25 (generatedRoot.w_1053_0 ctx)
  h1056_0 : MxxRuntime.concatRows (generatedRoot.w_297_6 ctx) (generatedRoot.w_297_8 ctx) (generatedRoot.w_1056_0 ctx)
  h1057_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1056_0 ctx) (generatedRoot.w_1057_0 ctx)
  h1058_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1057_0 ctx) (generatedRoot.w_1058_0 ctx)
  t1397 : 0 ≤ 1
  t1398 : 1 < 2
  t1399 : 2 ≤ 3
  t1400 : 0 ≤ 0
  t1401 : 0 < 25
  t1402 : 25 ≤ 25
  h1060_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1059_0 ctx) 1 2 0 25 (generatedRoot.w_1060_0 ctx)
  t1404 : 0 ≤ 2
  t1405 : 2 < 3
  t1406 : 3 ≤ 3
  t1407 : 0 ≤ 0

theorem generatedRoot.facts_43  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_43 backend (generatedRoot.w_297_6 ctx) (generatedRoot.w_297_8 ctx) (generatedRoot.w_1041_0 ctx) (generatedRoot.w_1047_0 ctx) (generatedRoot.w_1050_0 ctx) (generatedRoot.w_1052_0 ctx) (generatedRoot.w_1053_0 ctx) (generatedRoot.w_1056_0 ctx) (generatedRoot.w_1057_0 ctx) (generatedRoot.w_1058_0 ctx) (generatedRoot.w_1059_0 ctx) (generatedRoot.w_1060_0 ctx)) :
    generatedRoot.Facts_43 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_44  (ctx : generatedRootCtx) : Prop where
  t1408 : 0 < 25
  t1409 : 25 ≤ 25
  h1061_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1059_0 ctx) 2 3 0 25 (generatedRoot.w_1061_0 ctx)
  h1062_0 : MxxRuntime.concatRows (generatedRoot.w_1060_0 ctx) (generatedRoot.w_1061_0 ctx) (generatedRoot.w_1062_0 ctx)
  h1063_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1062_0 ctx) (generatedRoot.w_1063_0 ctx)
  h1064_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1063_0 ctx) (generatedRoot.w_1064_0 ctx)
  t1414 : 0 ≤ 1
  t1415 : 1 < 2
  t1416 : 2 ≤ 3
  t1417 : 0 ≤ 0
  t1418 : 0 < 25
  t1419 : 25 ≤ 25
  h1066_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1065_0 ctx) 1 2 0 25 (generatedRoot.w_1066_0 ctx)
  t1421 : 0 ≤ 2
  t1422 : 2 < 3
  t1423 : 3 ≤ 3
  t1424 : 0 ≤ 0
  t1425 : 0 < 25
  t1426 : 25 ≤ 25
  h1068_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1065_0 ctx) 2 3 0 25 (generatedRoot.w_1068_0 ctx)
  t1428 : 0 ≤ 0
  t1429 : 0 < 1
  t1430 : 1 ≤ 3
  t1431 : 0 ≤ 0
  t1432 : 0 < 25
  t1433 : 25 ≤ 25
  h1070_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1065_0 ctx) 0 1 0 25 (generatedRoot.w_1070_0 ctx)
  t1435 : 0 ≤ 0
  t1436 : 0 < 1
  t1437 : 1 ≤ 3
  t1438 : 0 ≤ 0
  t1439 : 0 < 25

theorem generatedRoot.facts_44  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_44 backend (generatedRoot.w_1059_0 ctx) (generatedRoot.w_1060_0 ctx) (generatedRoot.w_1061_0 ctx) (generatedRoot.w_1062_0 ctx) (generatedRoot.w_1063_0 ctx) (generatedRoot.w_1064_0 ctx) (generatedRoot.w_1065_0 ctx) (generatedRoot.w_1066_0 ctx) (generatedRoot.w_1068_0 ctx) (generatedRoot.w_1070_0 ctx)) :
    generatedRoot.Facts_44 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_45  (ctx : generatedRootCtx) : Prop where
  t1440 : 25 ≤ 25
  h1071_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1059_0 ctx) 0 1 0 25 (generatedRoot.w_1071_0 ctx)
  h1074_0 : MxxRuntime.concatRows (generatedRoot.w_355_6 ctx) (generatedRoot.w_355_8 ctx) (generatedRoot.w_1074_0 ctx)
  h1075_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1074_0 ctx) (generatedRoot.w_1075_0 ctx)
  h1076_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1075_0 ctx) (generatedRoot.w_1076_0 ctx)
  t1445 : 0 ≤ 1
  t1446 : 1 < 2
  t1447 : 2 ≤ 3
  t1448 : 0 ≤ 0
  t1449 : 0 < 25
  t1450 : 25 ≤ 25
  h1078_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1077_0 ctx) 1 2 0 25 (generatedRoot.w_1078_0 ctx)
  t1452 : 0 ≤ 2
  t1453 : 2 < 3
  t1454 : 3 ≤ 3
  t1455 : 0 ≤ 0
  t1456 : 0 < 25
  t1457 : 25 ≤ 25
  h1079_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1077_0 ctx) 2 3 0 25 (generatedRoot.w_1079_0 ctx)
  h1080_0 : MxxRuntime.concatRows (generatedRoot.w_1078_0 ctx) (generatedRoot.w_1079_0 ctx) (generatedRoot.w_1080_0 ctx)
  h1081_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1080_0 ctx) (generatedRoot.w_1081_0 ctx)
  h1082_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1081_0 ctx) (generatedRoot.w_1082_0 ctx)
  t1462 : 0 ≤ 1
  t1463 : 1 < 2
  t1464 : 2 ≤ 3
  t1465 : 0 ≤ 0
  t1466 : 0 < 25
  t1467 : 25 ≤ 25
  h1084_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1083_0 ctx) 1 2 0 25 (generatedRoot.w_1084_0 ctx)
  t1469 : 0 ≤ 2
  t1470 : 2 < 3
  t1471 : 3 ≤ 3

theorem generatedRoot.facts_45  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_45 backend (generatedRoot.w_355_6 ctx) (generatedRoot.w_355_8 ctx) (generatedRoot.w_1059_0 ctx) (generatedRoot.w_1071_0 ctx) (generatedRoot.w_1074_0 ctx) (generatedRoot.w_1075_0 ctx) (generatedRoot.w_1076_0 ctx) (generatedRoot.w_1077_0 ctx) (generatedRoot.w_1078_0 ctx) (generatedRoot.w_1079_0 ctx) (generatedRoot.w_1080_0 ctx) (generatedRoot.w_1081_0 ctx) (generatedRoot.w_1082_0 ctx) (generatedRoot.w_1083_0 ctx) (generatedRoot.w_1084_0 ctx)) :
    generatedRoot.Facts_45 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_46  (ctx : generatedRootCtx) : Prop where
  t1472 : 0 ≤ 0
  t1473 : 0 < 25
  t1474 : 25 ≤ 25
  h1086_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1083_0 ctx) 2 3 0 25 (generatedRoot.w_1086_0 ctx)
  t1476 : 0 ≤ 0
  t1477 : 0 < 1
  t1478 : 1 ≤ 3
  t1479 : 0 ≤ 0
  t1480 : 0 < 25
  t1481 : 25 ≤ 25
  h1088_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1083_0 ctx) 0 1 0 25 (generatedRoot.w_1088_0 ctx)
  t1483 : 0 ≤ 0
  t1484 : 0 < 1
  t1485 : 1 ≤ 3
  t1486 : 0 ≤ 0
  t1487 : 0 < 25
  t1488 : 25 ≤ 25
  h1089_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1077_0 ctx) 0 1 0 25 (generatedRoot.w_1089_0 ctx)
  h1092_0 : MxxRuntime.concatRows (generatedRoot.w_413_6 ctx) (generatedRoot.w_413_8 ctx) (generatedRoot.w_1092_0 ctx)
  h1093_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1092_0 ctx) (generatedRoot.w_1093_0 ctx)
  h1094_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1093_0 ctx) (generatedRoot.w_1094_0 ctx)
  t1493 : 0 ≤ 1
  t1494 : 1 < 2
  t1495 : 2 ≤ 3
  t1496 : 0 ≤ 0
  t1497 : 0 < 25
  t1498 : 25 ≤ 25
  h1096_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1095_0 ctx) 1 2 0 25 (generatedRoot.w_1096_0 ctx)
  t1500 : 0 ≤ 2
  t1501 : 2 < 3
  t1502 : 3 ≤ 3
  t1503 : 0 ≤ 0

theorem generatedRoot.facts_46  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_46 backend (generatedRoot.w_413_6 ctx) (generatedRoot.w_413_8 ctx) (generatedRoot.w_1077_0 ctx) (generatedRoot.w_1083_0 ctx) (generatedRoot.w_1086_0 ctx) (generatedRoot.w_1088_0 ctx) (generatedRoot.w_1089_0 ctx) (generatedRoot.w_1092_0 ctx) (generatedRoot.w_1093_0 ctx) (generatedRoot.w_1094_0 ctx) (generatedRoot.w_1095_0 ctx) (generatedRoot.w_1096_0 ctx)) :
    generatedRoot.Facts_46 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_47  (ctx : generatedRootCtx) : Prop where
  t1504 : 0 < 25
  t1505 : 25 ≤ 25
  h1097_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1095_0 ctx) 2 3 0 25 (generatedRoot.w_1097_0 ctx)
  h1098_0 : MxxRuntime.concatRows (generatedRoot.w_1096_0 ctx) (generatedRoot.w_1097_0 ctx) (generatedRoot.w_1098_0 ctx)
  h1099_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1098_0 ctx) (generatedRoot.w_1099_0 ctx)
  h1100_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1099_0 ctx) (generatedRoot.w_1100_0 ctx)
  t1510 : 0 ≤ 1
  t1511 : 1 < 2
  t1512 : 2 ≤ 3
  t1513 : 0 ≤ 0
  t1514 : 0 < 25
  t1515 : 25 ≤ 25
  h1102_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1101_0 ctx) 1 2 0 25 (generatedRoot.w_1102_0 ctx)
  t1517 : 0 ≤ 2
  t1518 : 2 < 3
  t1519 : 3 ≤ 3
  t1520 : 0 ≤ 0
  t1521 : 0 < 25
  t1522 : 25 ≤ 25
  h1104_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1101_0 ctx) 2 3 0 25 (generatedRoot.w_1104_0 ctx)
  t1524 : 0 ≤ 0
  t1525 : 0 < 1
  t1526 : 1 ≤ 3
  t1527 : 0 ≤ 0
  t1528 : 0 < 25
  t1529 : 25 ≤ 25
  h1106_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1101_0 ctx) 0 1 0 25 (generatedRoot.w_1106_0 ctx)
  t1531 : 0 ≤ 0
  t1532 : 0 < 1
  t1533 : 1 ≤ 3
  t1534 : 0 ≤ 0
  t1535 : 0 < 25

theorem generatedRoot.facts_47  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_47 backend (generatedRoot.w_1095_0 ctx) (generatedRoot.w_1096_0 ctx) (generatedRoot.w_1097_0 ctx) (generatedRoot.w_1098_0 ctx) (generatedRoot.w_1099_0 ctx) (generatedRoot.w_1100_0 ctx) (generatedRoot.w_1101_0 ctx) (generatedRoot.w_1102_0 ctx) (generatedRoot.w_1104_0 ctx) (generatedRoot.w_1106_0 ctx)) :
    generatedRoot.Facts_47 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_48  (ctx : generatedRootCtx) : Prop where
  t1536 : 25 ≤ 25
  h1107_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1095_0 ctx) 0 1 0 25 (generatedRoot.w_1107_0 ctx)
  h1110_0 : MxxRuntime.concatRows (generatedRoot.w_471_6 ctx) (generatedRoot.w_471_8 ctx) (generatedRoot.w_1110_0 ctx)
  h1111_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1110_0 ctx) (generatedRoot.w_1111_0 ctx)
  h1112_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1111_0 ctx) (generatedRoot.w_1112_0 ctx)
  t1541 : 0 ≤ 1
  t1542 : 1 < 2
  t1543 : 2 ≤ 3
  t1544 : 0 ≤ 0
  t1545 : 0 < 25
  t1546 : 25 ≤ 25
  h1114_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1113_0 ctx) 1 2 0 25 (generatedRoot.w_1114_0 ctx)
  t1548 : 0 ≤ 2
  t1549 : 2 < 3
  t1550 : 3 ≤ 3
  t1551 : 0 ≤ 0
  t1552 : 0 < 25
  t1553 : 25 ≤ 25
  h1115_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1113_0 ctx) 2 3 0 25 (generatedRoot.w_1115_0 ctx)
  h1116_0 : MxxRuntime.concatRows (generatedRoot.w_1114_0 ctx) (generatedRoot.w_1115_0 ctx) (generatedRoot.w_1116_0 ctx)
  h1117_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1116_0 ctx) (generatedRoot.w_1117_0 ctx)
  h1118_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1117_0 ctx) (generatedRoot.w_1118_0 ctx)
  t1558 : 0 ≤ 1
  t1559 : 1 < 2
  t1560 : 2 ≤ 3
  t1561 : 0 ≤ 0
  t1562 : 0 < 25
  t1563 : 25 ≤ 25
  h1120_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1119_0 ctx) 1 2 0 25 (generatedRoot.w_1120_0 ctx)
  t1565 : 0 ≤ 2
  t1566 : 2 < 3
  t1567 : 3 ≤ 3

theorem generatedRoot.facts_48  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_48 backend (generatedRoot.w_471_6 ctx) (generatedRoot.w_471_8 ctx) (generatedRoot.w_1095_0 ctx) (generatedRoot.w_1107_0 ctx) (generatedRoot.w_1110_0 ctx) (generatedRoot.w_1111_0 ctx) (generatedRoot.w_1112_0 ctx) (generatedRoot.w_1113_0 ctx) (generatedRoot.w_1114_0 ctx) (generatedRoot.w_1115_0 ctx) (generatedRoot.w_1116_0 ctx) (generatedRoot.w_1117_0 ctx) (generatedRoot.w_1118_0 ctx) (generatedRoot.w_1119_0 ctx) (generatedRoot.w_1120_0 ctx)) :
    generatedRoot.Facts_48 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_49  (ctx : generatedRootCtx) : Prop where
  t1568 : 0 ≤ 0
  t1569 : 0 < 25
  t1570 : 25 ≤ 25
  h1122_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1119_0 ctx) 2 3 0 25 (generatedRoot.w_1122_0 ctx)
  t1572 : 0 ≤ 0
  t1573 : 0 < 1
  t1574 : 1 ≤ 3
  t1575 : 0 ≤ 0
  t1576 : 0 < 25
  t1577 : 25 ≤ 25
  h1124_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1119_0 ctx) 0 1 0 25 (generatedRoot.w_1124_0 ctx)
  t1579 : 0 ≤ 0
  t1580 : 0 < 1
  t1581 : 1 ≤ 3
  t1582 : 0 ≤ 0
  t1583 : 0 < 25
  t1584 : 25 ≤ 25
  h1125_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1113_0 ctx) 0 1 0 25 (generatedRoot.w_1125_0 ctx)
  t1586 : (1) = 1
  h1128_0 : (∀ i : Fin 1, parallel_generatedRoot_1128 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1128_0 ctx) i))
  t1588 : (1) = 1
  h1129_0 : (∀ i : Fin 1, parallel_generatedRoot_1129 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1129_0 ctx) i))
  t1590 : (1) = 1
  h1130_0 : (∀ i : Fin 1, parallel_generatedRoot_1130 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1130_0 ctx) i))
  t1592 : (1) = 1
  h1131_0 : (∀ i : Fin 1, parallel_generatedRoot_1131 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1131_0 ctx) i))
  t1594 : (1) = 1
  h1132_0 : (∀ i : Fin 1, parallel_generatedRoot_1132 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1132_0 ctx) i))
  t1596 : (1) = 1
  h1133_0 : (∀ i : Fin 1, parallel_generatedRoot_1133 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1133_0 ctx) i))
  t1598 : (1) = 1
  h1134_0 : (∀ i : Fin 1, parallel_generatedRoot_1134 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1134_0 ctx) i))

theorem generatedRoot.facts_49  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_49 backend hashModel params (generatedRoot.w_991_0 ctx) (generatedRoot.w_1113_0 ctx) (generatedRoot.w_1119_0 ctx) (generatedRoot.w_1122_0 ctx) (generatedRoot.w_1124_0 ctx) (generatedRoot.w_1125_0 ctx) (generatedRoot.w_1128_0 ctx) (generatedRoot.w_1129_0 ctx) (generatedRoot.w_1130_0 ctx) (generatedRoot.w_1131_0 ctx) (generatedRoot.w_1132_0 ctx) (generatedRoot.w_1133_0 ctx) (generatedRoot.w_1134_0 ctx)) :
    generatedRoot.Facts_49 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_50  (ctx : generatedRootCtx) : Prop where
  t1600 : (1) = 1
  h1135_0 : (∀ i : Fin 1, parallel_generatedRoot_1135 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1135_0 ctx) i))
  t1602 : (1) = 1
  h1136_0 : (∀ i : Fin 1, parallel_generatedRoot_1136 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1136_0 ctx) i))
  t1604 : (1) = 1
  h1137_0 : (∀ i : Fin 1, parallel_generatedRoot_1137 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1137_0 ctx) i))
  t1606 : (1) = 1
  h1138_0 : (∀ i : Fin 1, parallel_generatedRoot_1138 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1138_0 ctx) i))
  t1608 : (1) = 1
  h1139_0 : (∀ i : Fin 1, parallel_generatedRoot_1139 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1139_0 ctx) i))
  t1610 : (1) = 1
  h1140_0 : (∀ i : Fin 1, parallel_generatedRoot_1140 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1140_0 ctx) i))
  t1612 : (1) = 1
  h1141_0 : (∀ i : Fin 1, parallel_generatedRoot_1141 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1141_0 ctx) i))
  t1614 : (8) = 8
  h1142_0 : (∀ i : Fin 8, parallel_generatedRoot_1142 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1142_0 ctx) i))
  t1616 : (8) = 8
  h1143_0 : (∀ i : Fin 8, parallel_generatedRoot_1143 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1143_0 ctx) i))
  t1618 : (4) = 4
  h1144_0 : (∀ i : Fin 4, parallel_generatedRoot_1144 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1144_0 ctx) i))
  t1620 : (4) = 4
  h1145_0 : (∀ i : Fin 4, parallel_generatedRoot_1145 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1145_0 ctx) i))
  t1622 : (2) = 2
  h1146_0 : (∀ i : Fin 2, parallel_generatedRoot_1146 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1146_0 ctx) i))
  t1624 : (2) = 2
  h1147_0 : (∀ i : Fin 2, parallel_generatedRoot_1147 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1147_0 ctx) i))
  t1626 : (4) = 4
  h1148_0 : (∀ i : Fin 4, parallel_generatedRoot_1148 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1148_0 ctx) i))
  t1628 : (4) = 4
  h1149_0 : (∀ i : Fin 4, parallel_generatedRoot_1149 backend hashModel params i (generatedRoot.w_991_0 ctx) ((generatedRoot.w_1149_0 ctx) i))
  h1150_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (0) (generatedRoot.w_1150_0 ctx)
  h1151_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (1) (generatedRoot.w_1151_0 ctx)

theorem generatedRoot.facts_50  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_50 backend hashModel params (generatedRoot.w_991_0 ctx) (generatedRoot.w_1135_0 ctx) (generatedRoot.w_1136_0 ctx) (generatedRoot.w_1137_0 ctx) (generatedRoot.w_1138_0 ctx) (generatedRoot.w_1139_0 ctx) (generatedRoot.w_1140_0 ctx) (generatedRoot.w_1141_0 ctx) (generatedRoot.w_1142_0 ctx) (generatedRoot.w_1143_0 ctx) (generatedRoot.w_1144_0 ctx) (generatedRoot.w_1145_0 ctx) (generatedRoot.w_1146_0 ctx) (generatedRoot.w_1147_0 ctx) (generatedRoot.w_1148_0 ctx) (generatedRoot.w_1149_0 ctx) (generatedRoot.w_1150_0 ctx) (generatedRoot.w_1151_0 ctx)) :
    generatedRoot.Facts_50 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_51  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  h1152_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (4) (generatedRoot.w_1152_0 ctx)
  h1153_0 : MxxRuntime.familyGetStatic (generatedRoot.w_991_0 ctx) (5) (generatedRoot.w_1153_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_51  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_51 (generatedRoot.w_21_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_56_2 ctx) (generatedRoot.w_56_3 ctx) (generatedRoot.w_56_4 ctx) (generatedRoot.w_56_5 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_330_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_446_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_562_0 ctx) (generatedRoot.w_620_0 ctx) (generatedRoot.w_654_0 ctx) (generatedRoot.w_657_0 ctx) (generatedRoot.w_664_0 ctx) (generatedRoot.w_667_0 ctx) (generatedRoot.w_673_0 ctx) (generatedRoot.w_678_0 ctx) (generatedRoot.w_683_0 ctx) (generatedRoot.w_688_0 ctx) (generatedRoot.w_693_0 ctx) (generatedRoot.w_698_0 ctx) (generatedRoot.w_703_0 ctx) (generatedRoot.w_711_0 ctx) (generatedRoot.w_714_0 ctx) (generatedRoot.w_720_0 ctx) (generatedRoot.w_725_0 ctx) (generatedRoot.w_730_0 ctx) (generatedRoot.w_735_0 ctx) (generatedRoot.w_740_0 ctx) (generatedRoot.w_745_0 ctx) (generatedRoot.w_750_0 ctx) (generatedRoot.w_929_0 ctx) (generatedRoot.w_929_1 ctx) (generatedRoot.w_929_2 ctx) (generatedRoot.w_929_3 ctx) (generatedRoot.w_929_4 ctx) (generatedRoot.w_929_5 ctx) (generatedRoot.w_983_0 ctx) (generatedRoot.w_983_1 ctx) (generatedRoot.w_983_2 ctx) (generatedRoot.w_983_3 ctx) (generatedRoot.w_983_4 ctx) (generatedRoot.w_983_5 ctx) (generatedRoot.w_991_0 ctx) (generatedRoot.w_997_0 ctx) (generatedRoot.w_1009_0 ctx) (generatedRoot.w_1011_0 ctx) (generatedRoot.w_1015_0 ctx) (generatedRoot.w_1027_0 ctx) (generatedRoot.w_1029_0 ctx) (generatedRoot.w_1033_0 ctx) (generatedRoot.w_1034_0 ctx) (generatedRoot.w_1035_0 ctx) (generatedRoot.w_1036_0 ctx) (generatedRoot.w_1037_0 ctx) (generatedRoot.w_1049_0 ctx) (generatedRoot.w_1051_0 ctx) (generatedRoot.w_1055_0 ctx) (generatedRoot.w_1067_0 ctx) (generatedRoot.w_1069_0 ctx) (generatedRoot.w_1073_0 ctx) (generatedRoot.w_1085_0 ctx) (generatedRoot.w_1087_0 ctx) (generatedRoot.w_1091_0 ctx) (generatedRoot.w_1103_0 ctx) (generatedRoot.w_1105_0 ctx) (generatedRoot.w_1109_0 ctx) (generatedRoot.w_1121_0 ctx) (generatedRoot.w_1123_0 ctx) (generatedRoot.w_1127_0 ctx) (generatedRoot.w_1128_0 ctx) (generatedRoot.w_1129_0 ctx) (generatedRoot.w_1130_0 ctx) (generatedRoot.w_1131_0 ctx) (generatedRoot.w_1132_0 ctx) (generatedRoot.w_1133_0 ctx) (generatedRoot.w_1134_0 ctx) (generatedRoot.w_1135_0 ctx) (generatedRoot.w_1136_0 ctx) (generatedRoot.w_1137_0 ctx) (generatedRoot.w_1138_0 ctx) (generatedRoot.w_1139_0 ctx) (generatedRoot.w_1140_0 ctx) (generatedRoot.w_1141_0 ctx) (generatedRoot.w_1142_0 ctx) (generatedRoot.w_1143_0 ctx) (generatedRoot.w_1144_0 ctx) (generatedRoot.w_1145_0 ctx) (generatedRoot.w_1146_0 ctx) (generatedRoot.w_1147_0 ctx) (generatedRoot.w_1148_0 ctx) (generatedRoot.w_1149_0 ctx) (generatedRoot.w_1150_0 ctx) (generatedRoot.w_1151_0 ctx) (generatedRoot.w_1152_0 ctx) (generatedRoot.w_1153_0 ctx) outputs) :
    generatedRoot.Facts_51 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

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
  c49 : generatedRoot.Facts_49 ctx
  c50 : generatedRoot.Facts_50 ctx
  c51 : generatedRoot.Facts_51 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body backend hashModel params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  ⟨(generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h.1.1.1.1.1), (generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) h.1.1.1.1.2.1), (generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) h.1.1.1.1.2.2), (generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) h.1.1.1.2.1), (generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) h.1.1.1.2.2.1), (generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) h.1.1.1.2.2.2), (generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) h.1.1.2.1.1), (generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) h.1.1.2.1.2.1), (generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) h.1.1.2.1.2.2), (generatedRoot.facts_9 (ctx := ctx) (outputs := outputs) h.1.1.2.2.1.1), (generatedRoot.facts_10 (ctx := ctx) (outputs := outputs) h.1.1.2.2.1.2), (generatedRoot.facts_11 (ctx := ctx) (outputs := outputs) h.1.1.2.2.2.1), (generatedRoot.facts_12 (ctx := ctx) (outputs := outputs) h.1.1.2.2.2.2), (generatedRoot.facts_13 (ctx := ctx) (outputs := outputs) h.1.2.1.1.1), (generatedRoot.facts_14 (ctx := ctx) (outputs := outputs) h.1.2.1.1.2.1), (generatedRoot.facts_15 (ctx := ctx) (outputs := outputs) h.1.2.1.1.2.2), (generatedRoot.facts_16 (ctx := ctx) (outputs := outputs) h.1.2.1.2.1), (generatedRoot.facts_17 (ctx := ctx) (outputs := outputs) h.1.2.1.2.2.1), (generatedRoot.facts_18 (ctx := ctx) (outputs := outputs) h.1.2.1.2.2.2), (generatedRoot.facts_19 (ctx := ctx) (outputs := outputs) h.1.2.2.1.1), (generatedRoot.facts_20 (ctx := ctx) (outputs := outputs) h.1.2.2.1.2.1), (generatedRoot.facts_21 (ctx := ctx) (outputs := outputs) h.1.2.2.1.2.2), (generatedRoot.facts_22 (ctx := ctx) (outputs := outputs) h.1.2.2.2.1.1), (generatedRoot.facts_23 (ctx := ctx) (outputs := outputs) h.1.2.2.2.1.2), (generatedRoot.facts_24 (ctx := ctx) (outputs := outputs) h.1.2.2.2.2.1), (generatedRoot.facts_25 (ctx := ctx) (outputs := outputs) h.1.2.2.2.2.2), (generatedRoot.facts_26 (ctx := ctx) (outputs := outputs) h.2.1.1.1.1), (generatedRoot.facts_27 (ctx := ctx) (outputs := outputs) h.2.1.1.1.2.1), (generatedRoot.facts_28 (ctx := ctx) (outputs := outputs) h.2.1.1.1.2.2), (generatedRoot.facts_29 (ctx := ctx) (outputs := outputs) h.2.1.1.2.1), (generatedRoot.facts_30 (ctx := ctx) (outputs := outputs) h.2.1.1.2.2.1), (generatedRoot.facts_31 (ctx := ctx) (outputs := outputs) h.2.1.1.2.2.2), (generatedRoot.facts_32 (ctx := ctx) (outputs := outputs) h.2.1.2.1.1), (generatedRoot.facts_33 (ctx := ctx) (outputs := outputs) h.2.1.2.1.2.1), (generatedRoot.facts_34 (ctx := ctx) (outputs := outputs) h.2.1.2.1.2.2), (generatedRoot.facts_35 (ctx := ctx) (outputs := outputs) h.2.1.2.2.1.1), (generatedRoot.facts_36 (ctx := ctx) (outputs := outputs) h.2.1.2.2.1.2), (generatedRoot.facts_37 (ctx := ctx) (outputs := outputs) h.2.1.2.2.2.1), (generatedRoot.facts_38 (ctx := ctx) (outputs := outputs) h.2.1.2.2.2.2), (generatedRoot.facts_39 (ctx := ctx) (outputs := outputs) h.2.2.1.1.1), (generatedRoot.facts_40 (ctx := ctx) (outputs := outputs) h.2.2.1.1.2.1), (generatedRoot.facts_41 (ctx := ctx) (outputs := outputs) h.2.2.1.1.2.2), (generatedRoot.facts_42 (ctx := ctx) (outputs := outputs) h.2.2.1.2.1), (generatedRoot.facts_43 (ctx := ctx) (outputs := outputs) h.2.2.1.2.2.1), (generatedRoot.facts_44 (ctx := ctx) (outputs := outputs) h.2.2.1.2.2.2), (generatedRoot.facts_45 (ctx := ctx) (outputs := outputs) h.2.2.2.1.1), (generatedRoot.facts_46 (ctx := ctx) (outputs := outputs) h.2.2.2.1.2.1), (generatedRoot.facts_47 (ctx := ctx) (outputs := outputs) h.2.2.2.1.2.2), (generatedRoot.facts_48 (ctx := ctx) (outputs := outputs) h.2.2.2.2.1.1), (generatedRoot.facts_49 (ctx := ctx) (outputs := outputs) h.2.2.2.2.1.2), (generatedRoot.facts_50 (ctx := ctx) (outputs := outputs) h.2.2.2.2.2.1), (generatedRoot.facts_51 (ctx := ctx) (outputs := outputs) h.2.2.2.2.2.2)⟩

theorem generatedRoot.h30_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_30.Witness, parallel_generatedRoot_30.Facts (⟨((generatedRoot.w_29_0 ctx), ((generatedRoot.w_29_1 ctx), ((generatedRoot.w_29_2 ctx), ((generatedRoot.w_29_3 ctx), ((generatedRoot.w_29_4 ctx), ((generatedRoot.w_29_5 ctx), ())))))), w⟩ : parallel_generatedRoot_30.Ctx backend hashModel params i) (((generatedRoot.w_30_0 ctx) i), (((generatedRoot.w_30_1 ctx) i), (((generatedRoot.w_30_2 ctx) i), (((generatedRoot.w_30_3 ctx) i), (((generatedRoot.w_30_4 ctx) i), (((generatedRoot.w_30_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h30_5 i
  ⟨w, parallel_generatedRoot_30.facts _ h⟩

theorem generatedRoot.h34_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_34.Witness, parallel_generatedRoot_34.Facts (⟨((generatedRoot.w_33_0 ctx), ((generatedRoot.w_33_1 ctx), ((generatedRoot.w_33_2 ctx), ((generatedRoot.w_33_3 ctx), ((generatedRoot.w_33_4 ctx), ((generatedRoot.w_33_5 ctx), ())))))), w⟩ : parallel_generatedRoot_34.Ctx backend hashModel params i) (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_1 ctx) i), (((generatedRoot.w_34_2 ctx) i), (((generatedRoot.w_34_3 ctx) i), (((generatedRoot.w_34_4 ctx) i), (((generatedRoot.w_34_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h34_5 i
  ⟨w, parallel_generatedRoot_34.facts _ h⟩

theorem generatedRoot.h35_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_35.Witness, parallel_generatedRoot_35.Facts (⟨((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_14_0 ctx), (((generatedRoot.w_34_4 ctx) i), (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_2 ctx) i), (((generatedRoot.w_34_5 ctx) i), (((generatedRoot.w_34_1 ctx) i), (((generatedRoot.w_34_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_35.Ctx backend hashModel params i) (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_35_1 ctx) i), (((generatedRoot.w_35_2 ctx) i), (((generatedRoot.w_35_3 ctx) i), (((generatedRoot.w_35_4 ctx) i), (((generatedRoot.w_35_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h35_5 i
  ⟨w, parallel_generatedRoot_35.facts _ h⟩

theorem generatedRoot.h36_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_36.Witness, parallel_generatedRoot_36.Facts (⟨((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), (((generatedRoot.w_35_4 ctx) i), (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_35_2 ctx) i), (((generatedRoot.w_35_5 ctx) i), (((generatedRoot.w_35_1 ctx) i), (((generatedRoot.w_35_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_36.Ctx backend hashModel params i) (((generatedRoot.w_36_0 ctx) i), (((generatedRoot.w_36_1 ctx) i), (((generatedRoot.w_36_2 ctx) i), (((generatedRoot.w_36_3 ctx) i), (((generatedRoot.w_36_4 ctx) i), (((generatedRoot.w_36_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h36_5 i
  ⟨w, parallel_generatedRoot_36.facts _ h⟩

theorem generatedRoot.h660_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 5) :
    ∃ w : parallel_generatedRoot_660.Witness, parallel_generatedRoot_660.Facts (⟨((generatedRoot.w_636_1 ctx), ((generatedRoot.w_638_0 ctx), ((generatedRoot.w_639_0 ctx), ((generatedRoot.w_636_0 ctx), ((generatedRoot.w_641_0 ctx), ((generatedRoot.w_642_0 ctx), ((generatedRoot.w_643_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_644_0 ctx), ((generatedRoot.w_645_0 ctx), ((generatedRoot.w_646_0 ctx), ((generatedRoot.w_647_0 ctx), ((generatedRoot.w_648_0 ctx), ((generatedRoot.w_649_0 ctx), ((generatedRoot.w_651_0 ctx), ((generatedRoot.w_653_0 ctx), ((generatedRoot.w_87_0 ctx), ((generatedRoot.w_656_0 ctx), ((generatedRoot.w_659_0 ctx), ((generatedRoot.w_88_0 ctx), ())))))))))))))))))))), w⟩ : parallel_generatedRoot_660.Ctx backend hashModel params i) (((generatedRoot.w_660_0 ctx) i), (((generatedRoot.w_660_1 ctx) i), (((generatedRoot.w_660_2 ctx) i), (((generatedRoot.w_660_3 ctx) i), (((generatedRoot.w_660_4 ctx) i), (((generatedRoot.w_660_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c27.h660_5 i
  ⟨w, parallel_generatedRoot_660.facts _ h⟩

theorem generatedRoot.h892_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_892.Witness, parallel_generatedRoot_892.Facts (⟨((generatedRoot.w_891_0 ctx), ((generatedRoot.w_891_1 ctx), ((generatedRoot.w_891_2 ctx), ((generatedRoot.w_891_3 ctx), ((generatedRoot.w_891_4 ctx), ((generatedRoot.w_891_5 ctx), ())))))), w⟩ : parallel_generatedRoot_892.Ctx backend hashModel params i) (((generatedRoot.w_892_0 ctx) i), (((generatedRoot.w_892_1 ctx) i), (((generatedRoot.w_892_2 ctx) i), (((generatedRoot.w_892_3 ctx) i), (((generatedRoot.w_892_4 ctx) i), (((generatedRoot.w_892_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c35.h892_5 i
  ⟨w, parallel_generatedRoot_892.facts _ h⟩

theorem generatedRoot.h896_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_896.Witness, parallel_generatedRoot_896.Facts (⟨((generatedRoot.w_895_0 ctx), ((generatedRoot.w_895_1 ctx), ((generatedRoot.w_895_2 ctx), ((generatedRoot.w_895_3 ctx), ((generatedRoot.w_895_4 ctx), ((generatedRoot.w_895_5 ctx), ())))))), w⟩ : parallel_generatedRoot_896.Ctx backend hashModel params i) (((generatedRoot.w_896_0 ctx) i), (((generatedRoot.w_896_1 ctx) i), (((generatedRoot.w_896_2 ctx) i), (((generatedRoot.w_896_3 ctx) i), (((generatedRoot.w_896_4 ctx) i), (((generatedRoot.w_896_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c35.h896_5 i
  ⟨w, parallel_generatedRoot_896.facts _ h⟩

theorem generatedRoot.h897_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_897.Witness, parallel_generatedRoot_897.Facts (⟨((generatedRoot.w_874_0 ctx), ((generatedRoot.w_875_0 ctx), ((generatedRoot.w_876_0 ctx), ((generatedRoot.w_877_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_14_0 ctx), (((generatedRoot.w_896_4 ctx) i), (((generatedRoot.w_896_0 ctx) i), (((generatedRoot.w_896_2 ctx) i), (((generatedRoot.w_896_5 ctx) i), (((generatedRoot.w_896_1 ctx) i), (((generatedRoot.w_896_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_897.Ctx backend hashModel params i) (((generatedRoot.w_897_0 ctx) i), (((generatedRoot.w_897_1 ctx) i), (((generatedRoot.w_897_2 ctx) i), (((generatedRoot.w_897_3 ctx) i), (((generatedRoot.w_897_4 ctx) i), (((generatedRoot.w_897_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c35.h897_5 i
  ⟨w, parallel_generatedRoot_897.facts _ h⟩

theorem generatedRoot.h946_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_946.Witness, parallel_generatedRoot_946.Facts (⟨((generatedRoot.w_945_0 ctx), ((generatedRoot.w_945_1 ctx), ((generatedRoot.w_945_2 ctx), ((generatedRoot.w_945_3 ctx), ((generatedRoot.w_945_4 ctx), ((generatedRoot.w_945_5 ctx), ())))))), w⟩ : parallel_generatedRoot_946.Ctx backend hashModel params i) (((generatedRoot.w_946_0 ctx) i), (((generatedRoot.w_946_1 ctx) i), (((generatedRoot.w_946_2 ctx) i), (((generatedRoot.w_946_3 ctx) i), (((generatedRoot.w_946_4 ctx) i), (((generatedRoot.w_946_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c37.h946_5 i
  ⟨w, parallel_generatedRoot_946.facts _ h⟩

theorem generatedRoot.h950_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_950.Witness, parallel_generatedRoot_950.Facts (⟨((generatedRoot.w_949_0 ctx), ((generatedRoot.w_949_1 ctx), ((generatedRoot.w_949_2 ctx), ((generatedRoot.w_949_3 ctx), ((generatedRoot.w_949_4 ctx), ((generatedRoot.w_949_5 ctx), ())))))), w⟩ : parallel_generatedRoot_950.Ctx backend hashModel params i) (((generatedRoot.w_950_0 ctx) i), (((generatedRoot.w_950_1 ctx) i), (((generatedRoot.w_950_2 ctx) i), (((generatedRoot.w_950_3 ctx) i), (((generatedRoot.w_950_4 ctx) i), (((generatedRoot.w_950_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c37.h950_5 i
  ⟨w, parallel_generatedRoot_950.facts _ h⟩

theorem generatedRoot.h951_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_951.Witness, parallel_generatedRoot_951.Facts (⟨((generatedRoot.w_931_0 ctx), ((generatedRoot.w_932_0 ctx), ((generatedRoot.w_933_0 ctx), ((generatedRoot.w_934_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_14_0 ctx), (((generatedRoot.w_950_4 ctx) i), (((generatedRoot.w_950_0 ctx) i), (((generatedRoot.w_950_2 ctx) i), (((generatedRoot.w_950_5 ctx) i), (((generatedRoot.w_950_1 ctx) i), (((generatedRoot.w_950_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_951.Ctx backend hashModel params i) (((generatedRoot.w_951_0 ctx) i), (((generatedRoot.w_951_1 ctx) i), (((generatedRoot.w_951_2 ctx) i), (((generatedRoot.w_951_3 ctx) i), (((generatedRoot.w_951_4 ctx) i), (((generatedRoot.w_951_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c37.h951_5 i
  ⟨w, parallel_generatedRoot_951.facts _ h⟩

theorem generatedRoot.h991_0_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36) :
    ∃ w : parallel_generatedRoot_991.Witness, parallel_generatedRoot_991.Facts (⟨((generatedRoot.w_0_0 ctx), (((generatedRoot.w_1_0 ctx) i), ((generatedRoot.w_63_0 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_872_3 ctx) i), (((generatedRoot.w_872_0 ctx) i), ((generatedRoot.w_929_5 ctx), ((generatedRoot.w_57_0 ctx), ((generatedRoot.w_983_5 ctx), ((generatedRoot.w_985_0 ctx), ((generatedRoot.w_983_1 ctx), ((generatedRoot.w_983_3 ctx), ((generatedRoot.w_986_0 ctx), (((generatedRoot.w_987_0 ctx) i), (((generatedRoot.w_988_0 ctx) i), (((generatedRoot.w_989_0 ctx) i), (((generatedRoot.w_990_0 ctx) i), ()))))))))))))))))), w⟩ : parallel_generatedRoot_991.Ctx backend hashModel params i) ((generatedRoot.w_991_0 ctx) i) :=
  let ⟨w, h⟩ := F.c38.h991_0 i
  ⟨w, parallel_generatedRoot_991.facts _ h⟩

end

end Stage_online_0

end
