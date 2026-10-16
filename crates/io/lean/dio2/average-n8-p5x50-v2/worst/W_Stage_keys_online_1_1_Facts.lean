import W_Stage_keys_online_1_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Stage_keys_online_1_1

local notation "mxxNat0" => 904625697163682872125158399399175198226681342373069820760217620276403972497
local notation "mxxNat1" => 36550556566445887949135852474979514307313305075533710697673106717124183540354
local notation "mxxNat2" => 84715014674746765542951856319120207364894094043218421459706511318570933778708
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 3 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 3 50
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 50 25
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 8 25 1
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 8 2 1

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_144.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit
  witness : parallel_generatedRoot_144.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_144Ctx" => (parallel_generatedRoot_144.Ctx backend hashModel params i_0)

def parallel_generatedRoot_144.w_0_0  (ctx : parallel_generatedRoot_144Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_144.w_1_0  (ctx : parallel_generatedRoot_144Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_144.w_2_0  (ctx : parallel_generatedRoot_144Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_144.w_3_0  (ctx : parallel_generatedRoot_144Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_144.w_4_0  (ctx : parallel_generatedRoot_144Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_144.w_5_0  (ctx : parallel_generatedRoot_144Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_144.w_6_0  (ctx : parallel_generatedRoot_144Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_144.w_6_1  (ctx : parallel_generatedRoot_144Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_144.w_6_2  (ctx : parallel_generatedRoot_144Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_144.w_6_3  (ctx : parallel_generatedRoot_144Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_144.w_6_4  (ctx : parallel_generatedRoot_144Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_144.w_6_5  (ctx : parallel_generatedRoot_144Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_144.w_7_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_144.w_8_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_144.w_9_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_144.w_7_0 ctx) (parallel_generatedRoot_144.w_8_0 ctx)

def parallel_generatedRoot_144.w_10_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_144.w_11_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_144.w_12_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_144.w_10_0 ctx) (parallel_generatedRoot_144.w_11_0 ctx)

def parallel_generatedRoot_144.w_13_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_144.w_14_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_144.w_15_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_144.w_13_0 ctx) (parallel_generatedRoot_144.w_14_0 ctx)

def parallel_generatedRoot_144.w_16_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_144.w_17_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_144.w_18_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_144.w_16_0 ctx) (parallel_generatedRoot_144.w_17_0 ctx)

def parallel_generatedRoot_144.w_19_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_144.w_20_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_144.w_21_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_144.w_19_0 ctx) (parallel_generatedRoot_144.w_20_0 ctx)

def parallel_generatedRoot_144.w_22_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_144.w_23_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_144.w_24_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_144.w_22_0 ctx) (parallel_generatedRoot_144.w_23_0 ctx)

def parallel_generatedRoot_144.w_25_0  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_144.w_25_1  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_144.w_25_2  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_144.w_25_3  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_144.w_25_4  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_144.w_25_5  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_144.outs  (ctx : parallel_generatedRoot_144Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_144.w_25_0 ctx), ((parallel_generatedRoot_144.w_25_1 ctx), ((parallel_generatedRoot_144.w_25_2 ctx), ((parallel_generatedRoot_144.w_25_3 ctx), ((parallel_generatedRoot_144.w_25_4 ctx), ((parallel_generatedRoot_144.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_144.Facts_0  (ctx : parallel_generatedRoot_144Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_144_6 backend hashModel params i_0 i ((parallel_generatedRoot_144.w_0_0 ctx), ((parallel_generatedRoot_144.w_1_0 ctx), ((parallel_generatedRoot_144.w_2_0 ctx), ((parallel_generatedRoot_144.w_3_0 ctx), ((parallel_generatedRoot_144.w_4_0 ctx), ((parallel_generatedRoot_144.w_5_0 ctx), ())))))) (((parallel_generatedRoot_144.w_6_0 ctx) i), (((parallel_generatedRoot_144.w_6_1 ctx) i), (((parallel_generatedRoot_144.w_6_2 ctx) i), (((parallel_generatedRoot_144.w_6_3 ctx) i), (((parallel_generatedRoot_144.w_6_4 ctx) i), (((parallel_generatedRoot_144.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_0 ctx) (0) (parallel_generatedRoot_144.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_0 ctx) (1) (parallel_generatedRoot_144.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_1 ctx) (0) (parallel_generatedRoot_144.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_1 ctx) (1) (parallel_generatedRoot_144.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_2 ctx) (0) (parallel_generatedRoot_144.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_2 ctx) (1) (parallel_generatedRoot_144.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_3 ctx) (0) (parallel_generatedRoot_144.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_3 ctx) (1) (parallel_generatedRoot_144.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_4 ctx) (0) (parallel_generatedRoot_144.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_4 ctx) (1) (parallel_generatedRoot_144.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_5 ctx) (0) (parallel_generatedRoot_144.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_144.w_6_5 ctx) (1) (parallel_generatedRoot_144.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_144_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_144.w_6_0 ctx), ((parallel_generatedRoot_144.w_6_1 ctx), ((parallel_generatedRoot_144.w_6_2 ctx), ((parallel_generatedRoot_144.w_6_3 ctx), ((parallel_generatedRoot_144.w_6_4 ctx), ((parallel_generatedRoot_144.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_144.w_9_0 ctx), ((parallel_generatedRoot_144.w_12_0 ctx), ((parallel_generatedRoot_144.w_15_0 ctx), ((parallel_generatedRoot_144.w_18_0 ctx), ((parallel_generatedRoot_144.w_21_0 ctx), ((parallel_generatedRoot_144.w_24_0 ctx), ())))))) ((parallel_generatedRoot_144.w_25_0 ctx), ((parallel_generatedRoot_144.w_25_1 ctx), ((parallel_generatedRoot_144.w_25_2 ctx), ((parallel_generatedRoot_144.w_25_3 ctx), ((parallel_generatedRoot_144.w_25_4 ctx), ((parallel_generatedRoot_144.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_144.outs ctx

theorem parallel_generatedRoot_144.facts_0  {ctx : parallel_generatedRoot_144Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_144.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_144.w_0_0 ctx) (parallel_generatedRoot_144.w_1_0 ctx) (parallel_generatedRoot_144.w_2_0 ctx) (parallel_generatedRoot_144.w_3_0 ctx) (parallel_generatedRoot_144.w_4_0 ctx) (parallel_generatedRoot_144.w_5_0 ctx) (parallel_generatedRoot_144.w_6_0 ctx) (parallel_generatedRoot_144.w_6_1 ctx) (parallel_generatedRoot_144.w_6_2 ctx) (parallel_generatedRoot_144.w_6_3 ctx) (parallel_generatedRoot_144.w_6_4 ctx) (parallel_generatedRoot_144.w_6_5 ctx) (parallel_generatedRoot_144.w_7_0 ctx) (parallel_generatedRoot_144.w_8_0 ctx) (parallel_generatedRoot_144.w_9_0 ctx) (parallel_generatedRoot_144.w_10_0 ctx) (parallel_generatedRoot_144.w_11_0 ctx) (parallel_generatedRoot_144.w_12_0 ctx) (parallel_generatedRoot_144.w_13_0 ctx) (parallel_generatedRoot_144.w_14_0 ctx) (parallel_generatedRoot_144.w_15_0 ctx) (parallel_generatedRoot_144.w_16_0 ctx) (parallel_generatedRoot_144.w_17_0 ctx) (parallel_generatedRoot_144.w_18_0 ctx) (parallel_generatedRoot_144.w_19_0 ctx) (parallel_generatedRoot_144.w_20_0 ctx) (parallel_generatedRoot_144.w_21_0 ctx) (parallel_generatedRoot_144.w_22_0 ctx) (parallel_generatedRoot_144.w_23_0 ctx) (parallel_generatedRoot_144.w_24_0 ctx) (parallel_generatedRoot_144.w_25_0 ctx) (parallel_generatedRoot_144.w_25_1 ctx) (parallel_generatedRoot_144.w_25_2 ctx) (parallel_generatedRoot_144.w_25_3 ctx) (parallel_generatedRoot_144.w_25_4 ctx) (parallel_generatedRoot_144.w_25_5 ctx) outputs) :
    parallel_generatedRoot_144.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_144.Facts  (ctx : parallel_generatedRoot_144Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_144.Facts_0 ctx outputs

theorem parallel_generatedRoot_144.facts  (ctx : parallel_generatedRoot_144Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_144.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_144.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_144.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_148.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit
  witness : parallel_generatedRoot_148.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_148Ctx" => (parallel_generatedRoot_148.Ctx backend hashModel params i_0)

def parallel_generatedRoot_148.w_0_0  (ctx : parallel_generatedRoot_148Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_148.w_1_0  (ctx : parallel_generatedRoot_148Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_148.w_2_0  (ctx : parallel_generatedRoot_148Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_148.w_3_0  (ctx : parallel_generatedRoot_148Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_148.w_4_0  (ctx : parallel_generatedRoot_148Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_148.w_5_0  (ctx : parallel_generatedRoot_148Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_148.w_6_0  (ctx : parallel_generatedRoot_148Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_148.w_6_1  (ctx : parallel_generatedRoot_148Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_148.w_6_2  (ctx : parallel_generatedRoot_148Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_148.w_6_3  (ctx : parallel_generatedRoot_148Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_148.w_6_4  (ctx : parallel_generatedRoot_148Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_148.w_6_5  (ctx : parallel_generatedRoot_148Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_148.w_7_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_148.w_8_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_148.w_9_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_148.w_7_0 ctx) (parallel_generatedRoot_148.w_8_0 ctx)

def parallel_generatedRoot_148.w_10_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_148.w_11_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_148.w_12_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_148.w_10_0 ctx) (parallel_generatedRoot_148.w_11_0 ctx)

def parallel_generatedRoot_148.w_13_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_148.w_14_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_148.w_15_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_148.w_13_0 ctx) (parallel_generatedRoot_148.w_14_0 ctx)

def parallel_generatedRoot_148.w_16_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_148.w_17_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_148.w_18_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_148.w_16_0 ctx) (parallel_generatedRoot_148.w_17_0 ctx)

def parallel_generatedRoot_148.w_19_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_148.w_20_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_148.w_21_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_148.w_19_0 ctx) (parallel_generatedRoot_148.w_20_0 ctx)

def parallel_generatedRoot_148.w_22_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_148.w_23_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_148.w_24_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_148.w_22_0 ctx) (parallel_generatedRoot_148.w_23_0 ctx)

def parallel_generatedRoot_148.w_25_0  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_148.w_25_1  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_148.w_25_2  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_148.w_25_3  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_148.w_25_4  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_148.w_25_5  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_148.outs  (ctx : parallel_generatedRoot_148Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_148.w_25_0 ctx), ((parallel_generatedRoot_148.w_25_1 ctx), ((parallel_generatedRoot_148.w_25_2 ctx), ((parallel_generatedRoot_148.w_25_3 ctx), ((parallel_generatedRoot_148.w_25_4 ctx), ((parallel_generatedRoot_148.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_148.Facts_0  (ctx : parallel_generatedRoot_148Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_148_6 backend hashModel params i_0 i ((parallel_generatedRoot_148.w_0_0 ctx), ((parallel_generatedRoot_148.w_1_0 ctx), ((parallel_generatedRoot_148.w_2_0 ctx), ((parallel_generatedRoot_148.w_3_0 ctx), ((parallel_generatedRoot_148.w_4_0 ctx), ((parallel_generatedRoot_148.w_5_0 ctx), ())))))) (((parallel_generatedRoot_148.w_6_0 ctx) i), (((parallel_generatedRoot_148.w_6_1 ctx) i), (((parallel_generatedRoot_148.w_6_2 ctx) i), (((parallel_generatedRoot_148.w_6_3 ctx) i), (((parallel_generatedRoot_148.w_6_4 ctx) i), (((parallel_generatedRoot_148.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_0 ctx) (0) (parallel_generatedRoot_148.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_0 ctx) (1) (parallel_generatedRoot_148.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_1 ctx) (0) (parallel_generatedRoot_148.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_1 ctx) (1) (parallel_generatedRoot_148.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_2 ctx) (0) (parallel_generatedRoot_148.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_2 ctx) (1) (parallel_generatedRoot_148.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_3 ctx) (0) (parallel_generatedRoot_148.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_3 ctx) (1) (parallel_generatedRoot_148.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_4 ctx) (0) (parallel_generatedRoot_148.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_4 ctx) (1) (parallel_generatedRoot_148.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_5 ctx) (0) (parallel_generatedRoot_148.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_148.w_6_5 ctx) (1) (parallel_generatedRoot_148.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_148_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_148.w_6_0 ctx), ((parallel_generatedRoot_148.w_6_1 ctx), ((parallel_generatedRoot_148.w_6_2 ctx), ((parallel_generatedRoot_148.w_6_3 ctx), ((parallel_generatedRoot_148.w_6_4 ctx), ((parallel_generatedRoot_148.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_148.w_9_0 ctx), ((parallel_generatedRoot_148.w_12_0 ctx), ((parallel_generatedRoot_148.w_15_0 ctx), ((parallel_generatedRoot_148.w_18_0 ctx), ((parallel_generatedRoot_148.w_21_0 ctx), ((parallel_generatedRoot_148.w_24_0 ctx), ())))))) ((parallel_generatedRoot_148.w_25_0 ctx), ((parallel_generatedRoot_148.w_25_1 ctx), ((parallel_generatedRoot_148.w_25_2 ctx), ((parallel_generatedRoot_148.w_25_3 ctx), ((parallel_generatedRoot_148.w_25_4 ctx), ((parallel_generatedRoot_148.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_148.outs ctx

theorem parallel_generatedRoot_148.facts_0  {ctx : parallel_generatedRoot_148Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_148.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_148.w_0_0 ctx) (parallel_generatedRoot_148.w_1_0 ctx) (parallel_generatedRoot_148.w_2_0 ctx) (parallel_generatedRoot_148.w_3_0 ctx) (parallel_generatedRoot_148.w_4_0 ctx) (parallel_generatedRoot_148.w_5_0 ctx) (parallel_generatedRoot_148.w_6_0 ctx) (parallel_generatedRoot_148.w_6_1 ctx) (parallel_generatedRoot_148.w_6_2 ctx) (parallel_generatedRoot_148.w_6_3 ctx) (parallel_generatedRoot_148.w_6_4 ctx) (parallel_generatedRoot_148.w_6_5 ctx) (parallel_generatedRoot_148.w_7_0 ctx) (parallel_generatedRoot_148.w_8_0 ctx) (parallel_generatedRoot_148.w_9_0 ctx) (parallel_generatedRoot_148.w_10_0 ctx) (parallel_generatedRoot_148.w_11_0 ctx) (parallel_generatedRoot_148.w_12_0 ctx) (parallel_generatedRoot_148.w_13_0 ctx) (parallel_generatedRoot_148.w_14_0 ctx) (parallel_generatedRoot_148.w_15_0 ctx) (parallel_generatedRoot_148.w_16_0 ctx) (parallel_generatedRoot_148.w_17_0 ctx) (parallel_generatedRoot_148.w_18_0 ctx) (parallel_generatedRoot_148.w_19_0 ctx) (parallel_generatedRoot_148.w_20_0 ctx) (parallel_generatedRoot_148.w_21_0 ctx) (parallel_generatedRoot_148.w_22_0 ctx) (parallel_generatedRoot_148.w_23_0 ctx) (parallel_generatedRoot_148.w_24_0 ctx) (parallel_generatedRoot_148.w_25_0 ctx) (parallel_generatedRoot_148.w_25_1 ctx) (parallel_generatedRoot_148.w_25_2 ctx) (parallel_generatedRoot_148.w_25_3 ctx) (parallel_generatedRoot_148.w_25_4 ctx) (parallel_generatedRoot_148.w_25_5 ctx) outputs) :
    parallel_generatedRoot_148.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_148.Facts  (ctx : parallel_generatedRoot_148Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_148.Facts_0 ctx outputs

theorem parallel_generatedRoot_148.facts  (ctx : parallel_generatedRoot_148Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_148.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_148.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_148.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_149.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_149.InputType)
  witness : parallel_generatedRoot_149.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_149Ctx" => (parallel_generatedRoot_149.Ctx backend hashModel params i_0)

def parallel_generatedRoot_149.w_1_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_149.w_3_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_149.w_5_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_149.w_7_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_149.w_9_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_149.w_10_0  (ctx : parallel_generatedRoot_149Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_149.w_11_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_149.w_12_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_149.w_13_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_149.w_14_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_149.w_15_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_149.w_16_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_149.w_17_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_149.w_0_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_149.w_2_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_149.w_4_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_149.w_6_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_149.w_8_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_149.w_18_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_149.w_18_1  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_149.w_18_2  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_149.w_18_3  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_149.w_18_4  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_149.w_18_5  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_149.w_18_6  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_149.w_18_7  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_149.w_18_8  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_149.w_18_9  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_149.w_19_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_149.w_20_0  (ctx : parallel_generatedRoot_149Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_149.w_21_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_149.w_22_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_149.w_23_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_149.w_23_concat_1  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_149.w_24_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_149.w_25_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_149.w_26_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_149.w_27_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_149.w_27_concat_1  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_149.w_28_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_149.w_29_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_149.w_30_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_149.w_31_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_149.w_32_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_149.w_32_concat_1  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_149.w_33_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_149.w_34_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_149.w_35_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_149.w_36_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_149.w_36_concat_1  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_149.w_37_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_149.w_38_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_149.w_39_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_149.w_40_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_149.w_41_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_149.w_37_0 ctx) (parallel_generatedRoot_149.w_40_0 ctx)

def parallel_generatedRoot_149.w_42_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_149.w_43_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_149.w_44_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_149.w_45_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_149.w_46_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_149.w_47_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_149.w_28_0 ctx) (parallel_generatedRoot_149.w_46_0 ctx)

def parallel_generatedRoot_149.w_48_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_149.w_49_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_18_2 ctx) (parallel_generatedRoot_149.w_48_0 ctx)

def parallel_generatedRoot_149.w_50_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_149.w_51_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_149.w_52_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_149.w_53_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_149.w_37_0 ctx) (parallel_generatedRoot_149.w_52_0 ctx)

def parallel_generatedRoot_149.w_54_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_149.w_55_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_149.w_56_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_149.w_57_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_149.w_58_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_149.w_59_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_149.w_28_0 ctx) (parallel_generatedRoot_149.w_58_0 ctx)

def parallel_generatedRoot_149.w_60_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_149.w_61_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_18_3 ctx) (parallel_generatedRoot_149.w_60_0 ctx)

def parallel_generatedRoot_149.w_62_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_149.w_63_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_18_4 ctx) (parallel_generatedRoot_149.w_62_0 ctx)

def parallel_generatedRoot_149.w_64_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_149.w_65_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_18_5 ctx) (parallel_generatedRoot_149.w_64_0 ctx)

def parallel_generatedRoot_149.w_66_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_149.w_67_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_149.w_68_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_66_0 ctx) (parallel_generatedRoot_149.w_67_0 ctx)

def parallel_generatedRoot_149.w_69_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_18_0 ctx) (parallel_generatedRoot_149.w_68_0 ctx)

def parallel_generatedRoot_149.w_70_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_149.w_71_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_149.w_72_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_70_0 ctx) (parallel_generatedRoot_149.w_71_0 ctx)

def parallel_generatedRoot_149.w_73_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_18_1 ctx) (parallel_generatedRoot_149.w_72_0 ctx)

def parallel_generatedRoot_149.outs  (ctx : parallel_generatedRoot_149Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_149.w_49_0 ctx), ((parallel_generatedRoot_149.w_61_0 ctx), ((parallel_generatedRoot_149.w_63_0 ctx), ((parallel_generatedRoot_149.w_65_0 ctx), ((parallel_generatedRoot_149.w_69_0 ctx), ((parallel_generatedRoot_149.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_149.Facts_0  (ctx : parallel_generatedRoot_149Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_1_0 ctx) (0) (parallel_generatedRoot_149.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_3_0 ctx) (0) (parallel_generatedRoot_149.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_5_0 ctx) (0) (parallel_generatedRoot_149.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_7_0 ctx) (0) (parallel_generatedRoot_149.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_149_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_149.w_9_0 ctx), ((parallel_generatedRoot_149.w_10_0 ctx), ((parallel_generatedRoot_149.w_11_0 ctx), ((parallel_generatedRoot_149.w_12_0 ctx), ((parallel_generatedRoot_149.w_13_0 ctx), ((parallel_generatedRoot_149.w_14_0 ctx), ((parallel_generatedRoot_149.w_15_0 ctx), ((parallel_generatedRoot_149.w_16_0 ctx), ((parallel_generatedRoot_149.w_17_0 ctx), ((parallel_generatedRoot_149.w_1_0 ctx), ((parallel_generatedRoot_149.w_3_0 ctx), ((parallel_generatedRoot_149.w_5_0 ctx), ((parallel_generatedRoot_149.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_149.w_0_0 ctx), ((parallel_generatedRoot_149.w_0_0 ctx), ((parallel_generatedRoot_149.w_0_0 ctx), ((parallel_generatedRoot_149.w_0_0 ctx), ((parallel_generatedRoot_149.w_0_0 ctx), ((parallel_generatedRoot_149.w_0_0 ctx), ((parallel_generatedRoot_149.w_2_0 ctx), ((parallel_generatedRoot_149.w_4_0 ctx), ((parallel_generatedRoot_149.w_6_0 ctx), ((parallel_generatedRoot_149.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_149.w_18_0 ctx), ((parallel_generatedRoot_149.w_18_1 ctx), ((parallel_generatedRoot_149.w_18_2 ctx), ((parallel_generatedRoot_149.w_18_3 ctx), ((parallel_generatedRoot_149.w_18_4 ctx), ((parallel_generatedRoot_149.w_18_5 ctx), ((parallel_generatedRoot_149.w_18_6 ctx), ((parallel_generatedRoot_149.w_18_7 ctx), ((parallel_generatedRoot_149.w_18_8 ctx), ((parallel_generatedRoot_149.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_9_0 ctx) (7) (parallel_generatedRoot_149.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_149.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_149.w_10_0 ctx)) (parallel_generatedRoot_149.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_149.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_149.w_10_0 ctx)) (parallel_generatedRoot_149.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_19_0 ctx) (parallel_generatedRoot_149.w_21_0 ctx) (parallel_generatedRoot_149.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_23_concat_1 ctx) (parallel_generatedRoot_149.w_22_0 ctx) (parallel_generatedRoot_149.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_11_0 ctx) (7) (parallel_generatedRoot_149.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_149.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_149.w_10_0 ctx)) (parallel_generatedRoot_149.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_149.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_149.w_10_0 ctx)) (parallel_generatedRoot_149.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_24_0 ctx) (parallel_generatedRoot_149.w_25_0 ctx) (parallel_generatedRoot_149.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_27_concat_1 ctx) (parallel_generatedRoot_149.w_26_0 ctx) (parallel_generatedRoot_149.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_149.w_23_0 ctx) (parallel_generatedRoot_149.w_27_0 ctx) (parallel_generatedRoot_149.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_149.w_12_0 ctx) (parallel_generatedRoot_149.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_149.w_13_0 ctx) (parallel_generatedRoot_149.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_149.w_14_0 ctx) (parallel_generatedRoot_149.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_29_0 ctx) (parallel_generatedRoot_149.w_30_0 ctx) (parallel_generatedRoot_149.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_32_concat_1 ctx) (parallel_generatedRoot_149.w_31_0 ctx) (parallel_generatedRoot_149.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_149.facts_0  {ctx : parallel_generatedRoot_149Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_149.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_149.w_1_0 ctx) (parallel_generatedRoot_149.w_3_0 ctx) (parallel_generatedRoot_149.w_5_0 ctx) (parallel_generatedRoot_149.w_7_0 ctx) (parallel_generatedRoot_149.w_9_0 ctx) (parallel_generatedRoot_149.w_10_0 ctx) (parallel_generatedRoot_149.w_11_0 ctx) (parallel_generatedRoot_149.w_12_0 ctx) (parallel_generatedRoot_149.w_13_0 ctx) (parallel_generatedRoot_149.w_14_0 ctx) (parallel_generatedRoot_149.w_15_0 ctx) (parallel_generatedRoot_149.w_16_0 ctx) (parallel_generatedRoot_149.w_17_0 ctx) (parallel_generatedRoot_149.w_0_0 ctx) (parallel_generatedRoot_149.w_2_0 ctx) (parallel_generatedRoot_149.w_4_0 ctx) (parallel_generatedRoot_149.w_6_0 ctx) (parallel_generatedRoot_149.w_8_0 ctx) (parallel_generatedRoot_149.w_18_0 ctx) (parallel_generatedRoot_149.w_18_1 ctx) (parallel_generatedRoot_149.w_18_2 ctx) (parallel_generatedRoot_149.w_18_3 ctx) (parallel_generatedRoot_149.w_18_4 ctx) (parallel_generatedRoot_149.w_18_5 ctx) (parallel_generatedRoot_149.w_18_6 ctx) (parallel_generatedRoot_149.w_18_7 ctx) (parallel_generatedRoot_149.w_18_8 ctx) (parallel_generatedRoot_149.w_18_9 ctx) (parallel_generatedRoot_149.w_19_0 ctx) (parallel_generatedRoot_149.w_20_0 ctx) (parallel_generatedRoot_149.w_21_0 ctx) (parallel_generatedRoot_149.w_22_0 ctx) (parallel_generatedRoot_149.w_23_0 ctx) (parallel_generatedRoot_149.w_23_concat_1 ctx) (parallel_generatedRoot_149.w_24_0 ctx) (parallel_generatedRoot_149.w_25_0 ctx) (parallel_generatedRoot_149.w_26_0 ctx) (parallel_generatedRoot_149.w_27_0 ctx) (parallel_generatedRoot_149.w_27_concat_1 ctx) (parallel_generatedRoot_149.w_28_0 ctx) (parallel_generatedRoot_149.w_29_0 ctx) (parallel_generatedRoot_149.w_30_0 ctx) (parallel_generatedRoot_149.w_31_0 ctx) (parallel_generatedRoot_149.w_32_0 ctx) (parallel_generatedRoot_149.w_32_concat_1 ctx)) :
    parallel_generatedRoot_149.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_149.Facts_1  (ctx : parallel_generatedRoot_149Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_149.w_15_0 ctx) (parallel_generatedRoot_149.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_149.w_16_0 ctx) (parallel_generatedRoot_149.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_149.w_17_0 ctx) (parallel_generatedRoot_149.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_33_0 ctx) (parallel_generatedRoot_149.w_34_0 ctx) (parallel_generatedRoot_149.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_36_concat_1 ctx) (parallel_generatedRoot_149.w_35_0 ctx) (parallel_generatedRoot_149.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_149.w_32_0 ctx) (parallel_generatedRoot_149.w_36_0 ctx) (parallel_generatedRoot_149.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_18_6 ctx) (parallel_generatedRoot_149.w_18_8 ctx) (parallel_generatedRoot_149.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_149.w_38_0 ctx) (parallel_generatedRoot_149.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_149.w_39_0 ctx) (parallel_generatedRoot_149.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_149.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_149.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_42_0 ctx) (parallel_generatedRoot_149.w_43_0 ctx) (parallel_generatedRoot_149.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_149.w_44_0 ctx) (parallel_generatedRoot_149.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_149.w_45_0 ctx) (parallel_generatedRoot_149.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_149.facts_1  {ctx : parallel_generatedRoot_149Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_149.constraints_1 backend (parallel_generatedRoot_149.w_15_0 ctx) (parallel_generatedRoot_149.w_16_0 ctx) (parallel_generatedRoot_149.w_17_0 ctx) (parallel_generatedRoot_149.w_18_6 ctx) (parallel_generatedRoot_149.w_18_8 ctx) (parallel_generatedRoot_149.w_32_0 ctx) (parallel_generatedRoot_149.w_33_0 ctx) (parallel_generatedRoot_149.w_34_0 ctx) (parallel_generatedRoot_149.w_35_0 ctx) (parallel_generatedRoot_149.w_36_0 ctx) (parallel_generatedRoot_149.w_36_concat_1 ctx) (parallel_generatedRoot_149.w_37_0 ctx) (parallel_generatedRoot_149.w_38_0 ctx) (parallel_generatedRoot_149.w_39_0 ctx) (parallel_generatedRoot_149.w_40_0 ctx) (parallel_generatedRoot_149.w_41_0 ctx) (parallel_generatedRoot_149.w_42_0 ctx) (parallel_generatedRoot_149.w_43_0 ctx) (parallel_generatedRoot_149.w_44_0 ctx) (parallel_generatedRoot_149.w_45_0 ctx) (parallel_generatedRoot_149.w_46_0 ctx)) :
    parallel_generatedRoot_149.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_149.Facts_2  (ctx : parallel_generatedRoot_149Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_149.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_18_7 ctx) (parallel_generatedRoot_149.w_18_9 ctx) (parallel_generatedRoot_149.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_149.w_50_0 ctx) (parallel_generatedRoot_149.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_149.w_51_0 ctx) (parallel_generatedRoot_149.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_149.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_149.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_149.w_54_0 ctx) (parallel_generatedRoot_149.w_55_0 ctx) (parallel_generatedRoot_149.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_149.w_56_0 ctx) (parallel_generatedRoot_149.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_149.w_57_0 ctx) (parallel_generatedRoot_149.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_149.w_60_0 ctx)

theorem parallel_generatedRoot_149.facts_2  {ctx : parallel_generatedRoot_149Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_149.constraints_2 backend (parallel_generatedRoot_149.w_18_7 ctx) (parallel_generatedRoot_149.w_18_9 ctx) (parallel_generatedRoot_149.w_47_0 ctx) (parallel_generatedRoot_149.w_48_0 ctx) (parallel_generatedRoot_149.w_50_0 ctx) (parallel_generatedRoot_149.w_51_0 ctx) (parallel_generatedRoot_149.w_52_0 ctx) (parallel_generatedRoot_149.w_53_0 ctx) (parallel_generatedRoot_149.w_54_0 ctx) (parallel_generatedRoot_149.w_55_0 ctx) (parallel_generatedRoot_149.w_56_0 ctx) (parallel_generatedRoot_149.w_57_0 ctx) (parallel_generatedRoot_149.w_58_0 ctx) (parallel_generatedRoot_149.w_59_0 ctx) (parallel_generatedRoot_149.w_60_0 ctx)) :
    parallel_generatedRoot_149.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_149.Facts_3  (ctx : parallel_generatedRoot_149Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_149.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_149.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_149.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_149.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_149.facts_3  {ctx : parallel_generatedRoot_149Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_149.constraints_3 (parallel_generatedRoot_149.w_41_0 ctx) (parallel_generatedRoot_149.w_47_0 ctx) (parallel_generatedRoot_149.w_59_0 ctx) (parallel_generatedRoot_149.w_62_0 ctx) (parallel_generatedRoot_149.w_64_0 ctx) (parallel_generatedRoot_149.w_66_0 ctx) (parallel_generatedRoot_149.w_67_0 ctx)) :
    parallel_generatedRoot_149.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_149.Facts_4  (ctx : parallel_generatedRoot_149Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_149.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_149.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_149.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_149.outs ctx

theorem parallel_generatedRoot_149.facts_4  {ctx : parallel_generatedRoot_149Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_149.constraints_4 (parallel_generatedRoot_149.w_49_0 ctx) (parallel_generatedRoot_149.w_53_0 ctx) (parallel_generatedRoot_149.w_59_0 ctx) (parallel_generatedRoot_149.w_61_0 ctx) (parallel_generatedRoot_149.w_63_0 ctx) (parallel_generatedRoot_149.w_65_0 ctx) (parallel_generatedRoot_149.w_69_0 ctx) (parallel_generatedRoot_149.w_70_0 ctx) (parallel_generatedRoot_149.w_71_0 ctx) (parallel_generatedRoot_149.w_73_0 ctx) outputs) :
    parallel_generatedRoot_149.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_149.Facts  (ctx : parallel_generatedRoot_149Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_149.Facts_0 ctx
  c1 : parallel_generatedRoot_149.Facts_1 ctx
  c2 : parallel_generatedRoot_149.Facts_2 ctx
  c3 : parallel_generatedRoot_149.Facts_3 ctx
  c4 : parallel_generatedRoot_149.Facts_4 ctx outputs

theorem parallel_generatedRoot_149.facts  (ctx : parallel_generatedRoot_149Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_149.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_149.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_149.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_149.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_149.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_149.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_149.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel params)

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.1

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_59_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_66_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_68_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_78_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_127_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_128_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_129_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_130_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_131_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_132_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_133_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_134_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_137_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_140_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_141_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_142_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_145_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_146_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_2_0

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_4_0

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_6_0

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_8_0

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx)

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx)

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_19_0

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx)

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_22_0

def generatedRoot.w_22_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_22_concat_1

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx)

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_27_0

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_30_0

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_30_0 ctx) (generatedRoot.w_31_0 ctx)

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_33_0

def generatedRoot.w_33_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_33_concat_1

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_34_0

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_35_0

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_36_0

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_37_0

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_38_0

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_39_0

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_34_0 ctx) (generatedRoot.w_39_0 ctx)

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_41_0

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_44_0

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : mxxMat5 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx)

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_42_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_41_0 ctx) (generatedRoot.w_47_0 ctx)

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_49_0

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_50_0

def generatedRoot.w_51_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_51_0

def generatedRoot.w_51_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_51_concat_1

def generatedRoot.w_52_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_52_0

def generatedRoot.w_53_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_53_0

def generatedRoot.w_54_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_54_0

def generatedRoot.w_55_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_55_0

def generatedRoot.w_56_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_56_0

def generatedRoot.w_57_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_34_0 ctx) (generatedRoot.w_56_0 ctx)

def generatedRoot.w_58_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_58_0

def generatedRoot.w_60_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_59_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_61_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_58_0 ctx) (generatedRoot.w_60_0 ctx)

def generatedRoot.w_62_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_62_0

def generatedRoot.w_63_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_63_0

def generatedRoot.w_64_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_64_0

def generatedRoot.w_64_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_64_concat_1

def generatedRoot.w_65_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_65_0

def generatedRoot.w_67_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_67_0

def generatedRoot.w_69_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_69_0

def generatedRoot.w_70_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_70_0

def generatedRoot.w_71_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_71_0

def generatedRoot.w_72_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_72_0

def generatedRoot.w_73_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_65_0 ctx) (generatedRoot.w_72_0 ctx)

def generatedRoot.w_74_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_74_0

def generatedRoot.w_75_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_75_0

def generatedRoot.w_76_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_76_0

def generatedRoot.w_77_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_77_0

def generatedRoot.w_79_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_79_0

def generatedRoot.w_80_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_80_0

def generatedRoot.w_81_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_81_0

def generatedRoot.w_82_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_82_0

def generatedRoot.w_83_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_65_0 ctx) (generatedRoot.w_82_0 ctx)

def generatedRoot.w_84_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_84_0

def generatedRoot.w_85_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_85_0

def generatedRoot.w_86_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_86_0

def generatedRoot.w_87_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_0

def generatedRoot.w_87_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_1

def generatedRoot.w_87_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_2

def generatedRoot.w_87_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_3

def generatedRoot.w_87_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_4

def generatedRoot.w_87_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_5

def generatedRoot.w_87_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_6

def generatedRoot.w_87_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_7

def generatedRoot.w_87_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_8

def generatedRoot.w_87_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_9

def generatedRoot.w_88_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_88_0

def generatedRoot.w_89_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (3) [(1), (3), (5), (15)])

def generatedRoot.w_90_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_90_0

def generatedRoot.w_91_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_91_0

def generatedRoot.w_92_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_92_0

def generatedRoot.w_92_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_92_concat_1

def generatedRoot.w_93_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_93_0

def generatedRoot.w_94_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_94_0

def generatedRoot.w_95_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_95_0

def generatedRoot.w_96_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_96_0

def generatedRoot.w_96_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_96_concat_1

def generatedRoot.w_97_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_97_0

def generatedRoot.w_98_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_98_0

def generatedRoot.w_99_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_99_0

def generatedRoot.w_100_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_100_0

def generatedRoot.w_101_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_101_0

def generatedRoot.w_101_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_101_concat_1

def generatedRoot.w_102_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_102_0

def generatedRoot.w_103_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_103_0

def generatedRoot.w_104_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_104_0

def generatedRoot.w_105_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_105_0

def generatedRoot.w_105_concat_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_105_concat_1

def generatedRoot.w_106_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_106_0

def generatedRoot.w_107_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_107_0

def generatedRoot.w_108_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_108_0

def generatedRoot.w_109_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_109_0

def generatedRoot.w_110_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_106_0 ctx) (generatedRoot.w_109_0 ctx)

def generatedRoot.w_111_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_111_0

def generatedRoot.w_112_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_112_0

def generatedRoot.w_113_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_113_0

def generatedRoot.w_114_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_114_0

def generatedRoot.w_115_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_115_0

def generatedRoot.w_116_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_97_0 ctx) (generatedRoot.w_115_0 ctx)

def generatedRoot.w_117_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_117_0

def generatedRoot.w_118_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_3 ctx) (generatedRoot.w_117_0 ctx)

def generatedRoot.w_119_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)

def generatedRoot.w_120_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_120_0

def generatedRoot.w_121_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def generatedRoot.w_122_0  (ctx : generatedRootCtx) : mxxMat5 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_120_0 ctx) (generatedRoot.w_121_0 ctx)

def generatedRoot.w_123_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_123_0

def generatedRoot.w_124_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_124_0

def generatedRoot.w_125_0  (ctx : generatedRootCtx) : mxxMat5 :=
  MxxRuntime.matrixMul (generatedRoot.w_118_0 ctx) (generatedRoot.w_124_0 ctx)

def generatedRoot.w_126_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (0 : mxxMat5)

def generatedRoot.w_135_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_135_0

def generatedRoot.w_136_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_136_0

def generatedRoot.w_138_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_138_0

def generatedRoot.w_139_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_139_0

def generatedRoot.w_143_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_143_0

def generatedRoot.w_143_1  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_143_1

def generatedRoot.w_143_2  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_143_2

def generatedRoot.w_143_3  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_143_3

def generatedRoot.w_143_4  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_143_4

def generatedRoot.w_143_5  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_143_5

def generatedRoot.w_144_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_144_0

def generatedRoot.w_144_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_144_1

def generatedRoot.w_144_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_144_2

def generatedRoot.w_144_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_144_3

def generatedRoot.w_144_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_144_4

def generatedRoot.w_144_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_144_5

def generatedRoot.w_147_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_147_0

def generatedRoot.w_147_1  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_147_1

def generatedRoot.w_147_2  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_147_2

def generatedRoot.w_147_3  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_147_3

def generatedRoot.w_147_4  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_147_4

def generatedRoot.w_147_5  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_147_5

def generatedRoot.w_148_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_148_0

def generatedRoot.w_148_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_148_1

def generatedRoot.w_148_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_148_2

def generatedRoot.w_148_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_148_3

def generatedRoot.w_148_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_148_4

def generatedRoot.w_148_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_148_5

def generatedRoot.w_149_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_149_0

def generatedRoot.w_149_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_149_1

def generatedRoot.w_149_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_149_2

def generatedRoot.w_149_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_149_3

def generatedRoot.w_149_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_149_4

def generatedRoot.w_149_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_149_5

def generatedRoot.w_150_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_150_0

def generatedRoot.w_150_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_150_1

def generatedRoot.w_150_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_150_2

def generatedRoot.w_150_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_150_3

def generatedRoot.w_150_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_150_4

def generatedRoot.w_150_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_150_5

def generatedRoot.w_151_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_151_0

def generatedRoot.w_152_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_152_0

def generatedRoot.w_153_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx)

def generatedRoot.w_154_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_154_0

def generatedRoot.w_155_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_155_0

def generatedRoot.w_156_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx)

def generatedRoot.w_157_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_157_0

def generatedRoot.w_158_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_158_0

def generatedRoot.w_159_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx)

def generatedRoot.w_160_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_160_0

def generatedRoot.w_161_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_0

def generatedRoot.w_162_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx)

def generatedRoot.w_163_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_163_0

def generatedRoot.w_164_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_164_0

def generatedRoot.w_165_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx)

def generatedRoot.w_166_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_166_0

def generatedRoot.w_167_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_167_0

def generatedRoot.w_168_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_166_0 ctx) (generatedRoot.w_167_0 ctx)

def generatedRoot.w_169_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_169_0

def generatedRoot.w_169_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_169_1

def generatedRoot.w_169_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_169_2

def generatedRoot.w_169_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_169_3

def generatedRoot.w_169_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_169_4

def generatedRoot.w_169_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_169_5

def generatedRoot.w_170_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat5)

def generatedRoot.w_171_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat5 :=
  ctx.witness.w_171_0

def generatedRoot.w_172_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_172_0

def generatedRoot.w_172_1  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_172_1

def generatedRoot.w_172_2  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_172_2

def generatedRoot.w_173_0  (ctx : generatedRootCtx) : mxxMat5 :=
  MxxRuntime.matrixAdd (generatedRoot.w_125_0 ctx) (generatedRoot.w_172_1 ctx)

def generatedRoot.w_174_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_174_0

def generatedRoot.w_175_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_5 ctx) (generatedRoot.w_174_0 ctx)

def generatedRoot.w_176_0  (ctx : generatedRootCtx) : mxxMat5 :=
  MxxRuntime.matrixMul (generatedRoot.w_175_0 ctx) (generatedRoot.w_124_0 ctx)

def generatedRoot.w_177_0  (ctx : generatedRootCtx) : mxxMat5 :=
  MxxRuntime.matrixAdd (generatedRoot.w_176_0 ctx) (generatedRoot.w_172_2 ctx)

def generatedRoot.w_178_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_178_0

def generatedRoot.w_179_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_179_0

def generatedRoot.w_180_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_180_0

def generatedRoot.w_181_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_181_0

def generatedRoot.w_182_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_106_0 ctx) (generatedRoot.w_181_0 ctx)

def generatedRoot.w_183_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_183_0

def generatedRoot.w_184_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_184_0

def generatedRoot.w_185_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_185_0

def generatedRoot.w_186_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_186_0

def generatedRoot.w_187_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_187_0

def generatedRoot.w_188_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixMul (generatedRoot.w_97_0 ctx) (generatedRoot.w_187_0 ctx)

def generatedRoot.w_189_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_189_0

def generatedRoot.w_190_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_2 ctx) (generatedRoot.w_189_0 ctx)

def generatedRoot.w_191_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_191_0

def generatedRoot.w_192_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_4 ctx) (generatedRoot.w_191_0 ctx)

def generatedRoot.w_193_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_193_0

def generatedRoot.w_194_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_194_0

def generatedRoot.w_195_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_193_0 ctx) (generatedRoot.w_194_0 ctx)

def generatedRoot.w_196_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_0 ctx) (generatedRoot.w_195_0 ctx)

def generatedRoot.w_197_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_197_0

def generatedRoot.w_198_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_198_0

def generatedRoot.w_199_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx)

def generatedRoot.w_200_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_1 ctx) (generatedRoot.w_199_0 ctx)

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_178_0 ctx), ((generatedRoot.w_190_0 ctx), ((generatedRoot.w_118_0 ctx), ((generatedRoot.w_192_0 ctx), ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_196_0 ctx), ((generatedRoot.w_200_0 ctx), ((generatedRoot.w_169_0 ctx), ((generatedRoot.w_169_1 ctx), ((generatedRoot.w_169_2 ctx), ((generatedRoot.w_169_3 ctx), ((generatedRoot.w_169_4 ctx), ((generatedRoot.w_169_5 ctx), ())))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (generatedRoot.w_1_0 ctx) (0) (generatedRoot.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (generatedRoot.w_3_0 ctx) (0) (generatedRoot.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (generatedRoot.w_5_0 ctx) (0) (generatedRoot.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (generatedRoot.w_7_0 ctx) (0) (generatedRoot.w_8_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (generatedRoot.w_12_0 ctx) (1) (generatedRoot.w_13_0 ctx)
  h16_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_16_0 ctx)
  h19_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_19_0 ctx)
  h18_0 : MxxRuntime.concatRows (generatedRoot.w_15_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_22_concat_1 ctx)
  h22_0 : MxxRuntime.concatRows (generatedRoot.w_22_concat_1 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (generatedRoot.w_23_0 ctx) (1) (generatedRoot.w_24_0 ctx)
  h27_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_27_0 ctx)
  h30_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_30_0 ctx)
  h29_0 : MxxRuntime.concatRows (generatedRoot.w_26_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_33_concat_1 ctx)
  h33_0 : MxxRuntime.concatRows (generatedRoot.w_33_concat_1 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx)
  h34_0 : MxxRuntime.concatColumns (generatedRoot.w_22_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx)
  h35_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_35_0 ctx)
  h36_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_36_0 ctx)
  h37_0 : MxxRuntime.concatRows (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx)
  h38_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx)
  h39_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx)
  t20 : 0 ≤ 0
  t21 : 0 < 1
  t22 : 1 ≤ 3
  t23 : 0 ≤ 0
  t24 : 0 < 25
  t25 : 25 ≤ 25
  h41_0 : MxxRuntime.sliceMatrix (generatedRoot.w_40_0 ctx) 0 1 0 25 (generatedRoot.w_41_0 ctx)
  h44_0 : MxxRuntime.familyGetStatic (generatedRoot.w_43_0 ctx) (1) (generatedRoot.w_44_0 ctx)
  t28 : 0 ≤ 1
  t29 : 1 < 2
  t30 : 2 ≤ 3
  t31 : 0 ≤ 0

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_22_concat_1 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_concat_1 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_44_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  t32 : 0 < 25
  t33 : 25 ≤ 25
  h49_0 : MxxRuntime.sliceMatrix (generatedRoot.w_40_0 ctx) 1 2 0 25 (generatedRoot.w_49_0 ctx)
  t35 : 0 ≤ 2
  t36 : 2 < 3
  t37 : 3 ≤ 3
  t38 : 0 ≤ 0
  t39 : 0 < 25
  t40 : 25 ≤ 25
  h50_0 : MxxRuntime.sliceMatrix (generatedRoot.w_40_0 ctx) 2 3 0 25 (generatedRoot.w_50_0 ctx)
  h49_0_1 : MxxRuntime.concatRows (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_concat_1 ctx)
  h51_0 : MxxRuntime.concatRows (generatedRoot.w_51_concat_1 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx)
  h52_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_52_0 ctx)
  h53_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_53_0 ctx)
  h54_0 : MxxRuntime.concatRows (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx)
  h55_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx)
  h56_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx)
  t49 : 0 ≤ 0
  t50 : 0 < 1
  t51 : 1 ≤ 3
  t52 : 0 ≤ 0
  t53 : 0 < 25
  t54 : 25 ≤ 25
  h58_0 : MxxRuntime.sliceMatrix (generatedRoot.w_57_0 ctx) 0 1 0 25 (generatedRoot.w_58_0 ctx)
  t56 : 0 ≤ 1
  t57 : 1 < 2
  t58 : 2 ≤ 3
  t59 : 0 ≤ 0
  t60 : 0 < 25
  t61 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (generatedRoot.w_57_0 ctx) 1 2 0 25 (generatedRoot.w_62_0 ctx)
  t63 : 0 ≤ 2

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_10_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_51_concat_1 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_62_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
  t64 : 2 < 3
  t65 : 3 ≤ 3
  t66 : 0 ≤ 0
  t67 : 0 < 25
  t68 : 25 ≤ 25
  h63_0 : MxxRuntime.sliceMatrix (generatedRoot.w_57_0 ctx) 2 3 0 25 (generatedRoot.w_63_0 ctx)
  h62_0_1 : MxxRuntime.concatRows (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_concat_1 ctx)
  h64_0 : MxxRuntime.concatRows (generatedRoot.w_64_concat_1 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx)
  h65_0 : MxxRuntime.concatColumns (generatedRoot.w_51_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_65_0 ctx)
  h67_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_66_0 ctx)) (generatedRoot.w_67_0 ctx)
  h69_0 : MxxRuntime.familyGetStatic (generatedRoot.w_68_0 ctx) (0) (generatedRoot.w_69_0 ctx)
  h70_0 : MxxRuntime.concatRows (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_70_0 ctx)
  h71_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx)
  h72_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx)
  t78 : 0 ≤ 0
  t79 : 0 < 1
  t80 : 1 ≤ 3
  t81 : 0 ≤ 0
  t82 : 0 < 25
  t83 : 25 ≤ 25
  h74_0 : MxxRuntime.sliceMatrix (generatedRoot.w_73_0 ctx) 0 1 0 25 (generatedRoot.w_74_0 ctx)
  t85 : 0 ≤ 1
  t86 : 1 < 2
  t87 : 2 ≤ 3
  t88 : 0 ≤ 0
  t89 : 0 < 25
  t90 : 25 ≤ 25
  h75_0 : MxxRuntime.sliceMatrix (generatedRoot.w_73_0 ctx) 1 2 0 25 (generatedRoot.w_75_0 ctx)
  t92 : 0 ≤ 2
  t93 : 2 < 3
  t94 : 3 ≤ 3
  t95 : 0 ≤ 0

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend hashModel (generatedRoot.w_66_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_64_concat_1 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
  t96 : 0 < 25
  t97 : 25 ≤ 25
  h76_0 : MxxRuntime.sliceMatrix (generatedRoot.w_73_0 ctx) 2 3 0 25 (generatedRoot.w_76_0 ctx)
  h77_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_66_0 ctx)) (generatedRoot.w_77_0 ctx)
  h79_0 : MxxRuntime.familyGetStatic (generatedRoot.w_78_0 ctx) (0) (generatedRoot.w_79_0 ctx)
  h80_0 : MxxRuntime.concatRows (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx)
  h81_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx)
  h82_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx)
  t104 : 0 ≤ 0
  t105 : 0 < 1
  t106 : 1 ≤ 3
  t107 : 0 ≤ 0
  t108 : 0 < 25
  t109 : 25 ≤ 25
  h84_0 : MxxRuntime.sliceMatrix (generatedRoot.w_83_0 ctx) 0 1 0 25 (generatedRoot.w_84_0 ctx)
  t111 : 0 ≤ 1
  t112 : 1 < 2
  t113 : 2 ≤ 3
  t114 : 0 ≤ 0
  t115 : 0 < 25
  t116 : 25 ≤ 25
  h85_0 : MxxRuntime.sliceMatrix (generatedRoot.w_83_0 ctx) 1 2 0 25 (generatedRoot.w_85_0 ctx)
  t118 : 0 ≤ 2
  t119 : 2 < 3
  t120 : 3 ≤ 3
  t121 : 0 ≤ 0
  t122 : 0 < 25
  t123 : 25 ≤ 25
  h86_0 : MxxRuntime.sliceMatrix (generatedRoot.w_83_0 ctx) 2 3 0 25 (generatedRoot.w_86_0 ctx)
  t125 : 0 ≤ (3)
  h87_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_87 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_74_0 ctx), ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_76_0 ctx), ((generatedRoot.w_84_0 ctx), ((generatedRoot.w_85_0 ctx), ((generatedRoot.w_86_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (3)) ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_8_0 ctx), ())))))))))) ((generatedRoot.w_87_0 ctx), ((generatedRoot.w_87_1 ctx), ((generatedRoot.w_87_2 ctx), ((generatedRoot.w_87_3 ctx), ((generatedRoot.w_87_4 ctx), ((generatedRoot.w_87_5 ctx), ((generatedRoot.w_87_6 ctx), ((generatedRoot.w_87_7 ctx), ((generatedRoot.w_87_8 ctx), ((generatedRoot.w_87_9 ctx), ()))))))))))
  h88_0 : MxxRuntime.familyGetStatic (generatedRoot.w_9_0 ctx) (3) (generatedRoot.w_88_0 ctx)

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_3 backend hashModel params (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_87_1 ctx) (generatedRoot.w_87_2 ctx) (generatedRoot.w_87_3 ctx) (generatedRoot.w_87_4 ctx) (generatedRoot.w_87_5 ctx) (generatedRoot.w_87_6 ctx) (generatedRoot.w_87_7 ctx) (generatedRoot.w_87_8 ctx) (generatedRoot.w_87_9 ctx) (generatedRoot.w_88_0 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4  (ctx : generatedRootCtx) : Prop where
  t128 : 0 ≤ (3)
  t129 : (3) < 4
  h90_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_89_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_90_0 ctx)
  h91_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_89_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_91_0 ctx)
  h90_0_1 : MxxRuntime.concatRows (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_concat_1 ctx)
  h92_0 : MxxRuntime.concatRows (generatedRoot.w_92_concat_1 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx)
  h93_0 : MxxRuntime.familyGetStatic (generatedRoot.w_11_0 ctx) (3) (generatedRoot.w_93_0 ctx)
  h94_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_89_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_94_0 ctx)
  h95_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_89_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_95_0 ctx)
  h94_0_1 : MxxRuntime.concatRows (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_concat_1 ctx)
  h96_0 : MxxRuntime.concatRows (generatedRoot.w_96_concat_1 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx)
  h97_0 : MxxRuntime.concatColumns (generatedRoot.w_92_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_97_0 ctx)
  t140 : 0 ≤ (3)
  t141 : (3) < 4
  h98_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_74_0 ctx) (generatedRoot.w_98_0 ctx)
  t143 : 0 ≤ (3)
  t144 : (3) < 4
  h99_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_75_0 ctx) (generatedRoot.w_99_0 ctx)
  t146 : 0 ≤ (3)
  t147 : (3) < 4
  h100_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_76_0 ctx) (generatedRoot.w_100_0 ctx)
  h99_0_1 : MxxRuntime.concatRows (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_concat_1 ctx)
  h101_0 : MxxRuntime.concatRows (generatedRoot.w_101_concat_1 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx)
  t151 : 0 ≤ (3)
  t152 : (3) < 4
  h102_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_84_0 ctx) (generatedRoot.w_102_0 ctx)
  t154 : 0 ≤ (3)
  t155 : (3) < 4
  h103_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_85_0 ctx) (generatedRoot.w_103_0 ctx)
  t157 : 0 ≤ (3)
  t158 : (3) < 4
  h104_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_86_0 ctx) (generatedRoot.w_104_0 ctx)

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_4 hashModel (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_92_concat_1 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_96_concat_1 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_101_concat_1 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
  h103_0_1 : MxxRuntime.concatRows (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_concat_1 ctx)
  h105_0 : MxxRuntime.concatRows (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx)
  h106_0 : MxxRuntime.concatColumns (generatedRoot.w_101_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_106_0 ctx)
  h107_0 : MxxRuntime.concatRows (generatedRoot.w_87_7 ctx) (generatedRoot.w_87_9 ctx) (generatedRoot.w_107_0 ctx)
  h108_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx)
  h109_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx)
  t166 : 0 ≤ 1
  t167 : 1 < 2
  t168 : 2 ≤ 3
  t169 : 0 ≤ 0
  t170 : 0 < 25
  t171 : 25 ≤ 25
  h111_0 : MxxRuntime.sliceMatrix (generatedRoot.w_110_0 ctx) 1 2 0 25 (generatedRoot.w_111_0 ctx)
  t173 : 0 ≤ 2
  t174 : 2 < 3
  t175 : 3 ≤ 3
  t176 : 0 ≤ 0
  t177 : 0 < 25
  t178 : 25 ≤ 25
  h112_0 : MxxRuntime.sliceMatrix (generatedRoot.w_110_0 ctx) 2 3 0 25 (generatedRoot.w_112_0 ctx)
  h113_0 : MxxRuntime.concatRows (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx)
  h114_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx)
  h115_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx)
  t183 : 0 ≤ 1
  t184 : 1 < 2
  t185 : 2 ≤ 3
  t186 : 0 ≤ 0
  t187 : 0 < 25
  t188 : 25 ≤ 25
  h117_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 1 2 0 25 (generatedRoot.w_117_0 ctx)
  h120_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx)
  h123_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx)

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_5 backend (generatedRoot.w_87_7 ctx) (generatedRoot.w_87_9 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
  h124_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx)
  h135_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_135_0 ctx)
  h136_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_136_0 ctx)
  h138_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_138_0 ctx)
  h139_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_139_0 ctx)
  t197 : (16) = 16
  h143_5 : (∀ i : Fin 16, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_134_0 ctx) i) ∧ ((generatedRoot.w_134_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_134_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_134_0 ctx) i) ∧ ((generatedRoot.w_134_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_134_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_134_0 ctx) i) ∧ ((generatedRoot.w_134_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_134_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_143 backend hashModel params i (((generatedRoot.w_134_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_135_0 ctx), ((generatedRoot.w_136_0 ctx), ((generatedRoot.w_137_0 ctx), (w_43_0_gathered, ((generatedRoot.w_138_0 ctx), ((generatedRoot.w_139_0 ctx), ((generatedRoot.w_140_0 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_141_0 ctx) i), (((generatedRoot.w_142_0 ctx) i), ())))))))))))))) (((generatedRoot.w_143_0 ctx) i), (((generatedRoot.w_143_1 ctx) i), (((generatedRoot.w_143_2 ctx) i), (((generatedRoot.w_143_3 ctx) i), (((generatedRoot.w_143_4 ctx) i), (((generatedRoot.w_143_5 ctx) i), ())))))))
  t199 : (8) = 8
  h144_5 : (∀ i : Fin 8, parallel_generatedRoot_144 backend hashModel params i ((generatedRoot.w_143_0 ctx), ((generatedRoot.w_143_1 ctx), ((generatedRoot.w_143_2 ctx), ((generatedRoot.w_143_3 ctx), ((generatedRoot.w_143_4 ctx), ((generatedRoot.w_143_5 ctx), ())))))) (((generatedRoot.w_144_0 ctx) i), (((generatedRoot.w_144_1 ctx) i), (((generatedRoot.w_144_2 ctx) i), (((generatedRoot.w_144_3 ctx) i), (((generatedRoot.w_144_4 ctx) i), (((generatedRoot.w_144_5 ctx) i), ())))))))
  t201 : (16) = 16
  h147_5 : (∀ i : Fin 16, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_133_0 ctx) i) ∧ ((generatedRoot.w_133_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_133_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_133_0 ctx) i) ∧ ((generatedRoot.w_133_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_133_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_133_0 ctx) i) ∧ ((generatedRoot.w_133_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_133_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_147 backend hashModel params i (((generatedRoot.w_133_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_144_0 ctx), ((generatedRoot.w_144_2 ctx), ((generatedRoot.w_144_4 ctx), (w_43_0_gathered, ((generatedRoot.w_144_1 ctx), ((generatedRoot.w_144_3 ctx), ((generatedRoot.w_144_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_145_0 ctx) i), (((generatedRoot.w_146_0 ctx) i), ())))))))))))))) (((generatedRoot.w_147_0 ctx) i), (((generatedRoot.w_147_1 ctx) i), (((generatedRoot.w_147_2 ctx) i), (((generatedRoot.w_147_3 ctx) i), (((generatedRoot.w_147_4 ctx) i), (((generatedRoot.w_147_5 ctx) i), ())))))))
  t203 : (8) = 8
  h148_5 : (∀ i : Fin 8, parallel_generatedRoot_148 backend hashModel params i ((generatedRoot.w_147_0 ctx), ((generatedRoot.w_147_1 ctx), ((generatedRoot.w_147_2 ctx), ((generatedRoot.w_147_3 ctx), ((generatedRoot.w_147_4 ctx), ((generatedRoot.w_147_5 ctx), ())))))) (((generatedRoot.w_148_0 ctx) i), (((generatedRoot.w_148_1 ctx) i), (((generatedRoot.w_148_2 ctx) i), (((generatedRoot.w_148_3 ctx) i), (((generatedRoot.w_148_4 ctx) i), (((generatedRoot.w_148_5 ctx) i), ())))))))
  t205 : (8) = 8
  h149_5 : (∀ i : Fin 8, parallel_generatedRoot_149 backend hashModel params i ((generatedRoot.w_127_0 ctx), ((generatedRoot.w_128_0 ctx), ((generatedRoot.w_129_0 ctx), ((generatedRoot.w_130_0 ctx), ((generatedRoot.w_131_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_132_0 ctx), (((generatedRoot.w_148_4 ctx) i), (((generatedRoot.w_148_0 ctx) i), (((generatedRoot.w_148_2 ctx) i), (((generatedRoot.w_148_5 ctx) i), (((generatedRoot.w_148_1 ctx) i), (((generatedRoot.w_148_3 ctx) i), ()))))))))))))) (((generatedRoot.w_149_0 ctx) i), (((generatedRoot.w_149_1 ctx) i), (((generatedRoot.w_149_2 ctx) i), (((generatedRoot.w_149_3 ctx) i), (((generatedRoot.w_149_4 ctx) i), (((generatedRoot.w_149_5 ctx) i), ())))))))
  t207 : (8) = 8
  h150_5 : (∀ i : Fin 8, parallel_generatedRoot_150 backend hashModel params i ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_149_1 ctx), ((generatedRoot.w_149_2 ctx), ((generatedRoot.w_149_3 ctx), ((generatedRoot.w_149_4 ctx), ((generatedRoot.w_149_5 ctx), ())))))) (((generatedRoot.w_150_0 ctx) i), (((generatedRoot.w_150_1 ctx) i), (((generatedRoot.w_150_2 ctx) i), (((generatedRoot.w_150_3 ctx) i), (((generatedRoot.w_150_4 ctx) i), (((generatedRoot.w_150_5 ctx) i), ())))))))
  h151_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_0 ctx) (0) (generatedRoot.w_151_0 ctx)
  h152_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_0 ctx) (1) (generatedRoot.w_152_0 ctx)
  h154_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_1 ctx) (0) (generatedRoot.w_154_0 ctx)
  h155_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_1 ctx) (1) (generatedRoot.w_155_0 ctx)
  h157_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_2 ctx) (0) (generatedRoot.w_157_0 ctx)
  h158_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_2 ctx) (1) (generatedRoot.w_158_0 ctx)
  h160_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_3 ctx) (0) (generatedRoot.w_160_0 ctx)
  h161_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_3 ctx) (1) (generatedRoot.w_161_0 ctx)
  h163_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_4 ctx) (0) (generatedRoot.w_163_0 ctx)
  h164_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_4 ctx) (1) (generatedRoot.w_164_0 ctx)
  h166_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_5 ctx) (0) (generatedRoot.w_166_0 ctx)
  h167_0 : MxxRuntime.familyGetStatic (generatedRoot.w_150_5 ctx) (1) (generatedRoot.w_167_0 ctx)
  t221 : 0 ≤ (6)
  h169_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_169 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_150_0 ctx), ((generatedRoot.w_150_1 ctx), ((generatedRoot.w_150_2 ctx), ((generatedRoot.w_150_3 ctx), ((generatedRoot.w_150_4 ctx), ((generatedRoot.w_150_5 ctx), ())))))))))))) next) (Int.toNat (6)) ((generatedRoot.w_153_0 ctx), ((generatedRoot.w_156_0 ctx), ((generatedRoot.w_159_0 ctx), ((generatedRoot.w_162_0 ctx), ((generatedRoot.w_165_0 ctx), ((generatedRoot.w_168_0 ctx), ())))))) ((generatedRoot.w_169_0 ctx), ((generatedRoot.w_169_1 ctx), ((generatedRoot.w_169_2 ctx), ((generatedRoot.w_169_3 ctx), ((generatedRoot.w_169_4 ctx), ((generatedRoot.w_169_5 ctx), ()))))))
  t223 : (1) = 1

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_6 backend hashModel params (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_143_1 ctx) (generatedRoot.w_143_2 ctx) (generatedRoot.w_143_3 ctx) (generatedRoot.w_143_4 ctx) (generatedRoot.w_143_5 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_144_1 ctx) (generatedRoot.w_144_2 ctx) (generatedRoot.w_144_3 ctx) (generatedRoot.w_144_4 ctx) (generatedRoot.w_144_5 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_147_1 ctx) (generatedRoot.w_147_2 ctx) (generatedRoot.w_147_3 ctx) (generatedRoot.w_147_4 ctx) (generatedRoot.w_147_5 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_148_1 ctx) (generatedRoot.w_148_2 ctx) (generatedRoot.w_148_3 ctx) (generatedRoot.w_148_4 ctx) (generatedRoot.w_148_5 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_149_1 ctx) (generatedRoot.w_149_2 ctx) (generatedRoot.w_149_3 ctx) (generatedRoot.w_149_4 ctx) (generatedRoot.w_149_5 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_150_1 ctx) (generatedRoot.w_150_2 ctx) (generatedRoot.w_150_3 ctx) (generatedRoot.w_150_4 ctx) (generatedRoot.w_150_5 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_169_1 ctx) (generatedRoot.w_169_2 ctx) (generatedRoot.w_169_3 ctx) (generatedRoot.w_169_4 ctx) (generatedRoot.w_169_5 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) : Prop where
  h171_0 : (∀ i : Fin 1, parallel_generatedRoot_171 backend hashModel params i (generatedRoot.w_170_0 ctx) ((generatedRoot.w_171_0 ctx) i))
  t225 : 0 ≤ (1)
  h172_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat5 × mxxMat5 × mxxMat5 × Unit) => sequential_generatedRoot_172 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, ((generatedRoot.w_169_5 ctx), ((generatedRoot.w_171_0 ctx), ((generatedRoot.w_169_1 ctx), ((generatedRoot.w_169_3 ctx), ()))))))) next) (Int.toNat (1)) ((generatedRoot.w_126_0 ctx), ((generatedRoot.w_126_0 ctx), ((generatedRoot.w_126_0 ctx), ()))) ((generatedRoot.w_172_0 ctx), ((generatedRoot.w_172_1 ctx), ((generatedRoot.w_172_2 ctx), ())))
  t227 : 0 ≤ 2
  t228 : 2 < 3
  t229 : 3 ≤ 3
  t230 : 0 ≤ 0
  t231 : 0 < 25
  t232 : 25 ≤ 25
  h174_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 2 3 0 25 (generatedRoot.w_174_0 ctx)
  h178_0 : MxxRuntime.concatRows (generatedRoot.w_173_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx)
  h179_0 : MxxRuntime.concatRows (generatedRoot.w_87_6 ctx) (generatedRoot.w_87_8 ctx) (generatedRoot.w_179_0 ctx)
  h180_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx)
  h181_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx)
  t238 : 0 ≤ 1
  t239 : 1 < 2
  t240 : 2 ≤ 3
  t241 : 0 ≤ 0
  t242 : 0 < 25
  t243 : 25 ≤ 25
  h183_0 : MxxRuntime.sliceMatrix (generatedRoot.w_182_0 ctx) 1 2 0 25 (generatedRoot.w_183_0 ctx)
  t245 : 0 ≤ 2
  t246 : 2 < 3
  t247 : 3 ≤ 3
  t248 : 0 ≤ 0
  t249 : 0 < 25
  t250 : 25 ≤ 25
  h184_0 : MxxRuntime.sliceMatrix (generatedRoot.w_182_0 ctx) 2 3 0 25 (generatedRoot.w_184_0 ctx)
  h185_0 : MxxRuntime.concatRows (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx)
  h186_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx)
  h187_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx)
  t255 : 0 ≤ 1

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_7 backend hashModel params (generatedRoot.w_87_6 ctx) (generatedRoot.w_87_8 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_169_1 ctx) (generatedRoot.w_169_3 ctx) (generatedRoot.w_169_5 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_172_1 ctx) (generatedRoot.w_172_2 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8  (ctx : generatedRootCtx) : Prop where
  t256 : 1 < 2
  t257 : 2 ≤ 3
  t258 : 0 ≤ 0
  t259 : 0 < 25
  t260 : 25 ≤ 25
  h189_0 : MxxRuntime.sliceMatrix (generatedRoot.w_188_0 ctx) 1 2 0 25 (generatedRoot.w_189_0 ctx)
  t262 : 0 ≤ 2
  t263 : 2 < 3
  t264 : 3 ≤ 3
  t265 : 0 ≤ 0
  t266 : 0 < 25
  t267 : 25 ≤ 25
  h191_0 : MxxRuntime.sliceMatrix (generatedRoot.w_188_0 ctx) 2 3 0 25 (generatedRoot.w_191_0 ctx)
  t269 : 0 ≤ 0
  t270 : 0 < 1
  t271 : 1 ≤ 3
  t272 : 0 ≤ 0
  t273 : 0 < 25
  t274 : 25 ≤ 25
  h193_0 : MxxRuntime.sliceMatrix (generatedRoot.w_188_0 ctx) 0 1 0 25 (generatedRoot.w_193_0 ctx)
  t276 : 0 ≤ 0
  t277 : 0 < 1
  t278 : 1 ≤ 3
  t279 : 0 ≤ 0
  t280 : 0 < 25
  t281 : 25 ≤ 25
  h194_0 : MxxRuntime.sliceMatrix (generatedRoot.w_182_0 ctx) 0 1 0 25 (generatedRoot.w_194_0 ctx)
  t283 : 0 ≤ 0
  t284 : 0 < 1
  t285 : 1 ≤ 3
  t286 : 0 ≤ 0
  t287 : 0 < 25

theorem generatedRoot.facts_8  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_8 (generatedRoot.w_182_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_194_0 ctx)) :
    generatedRoot.Facts_8 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_9  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t288 : 25 ≤ 25
  h197_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 0 1 0 25 (generatedRoot.w_197_0 ctx)
  t290 : 0 ≤ 0
  t291 : 0 < 1
  t292 : 1 ≤ 3
  t293 : 0 ≤ 0
  t294 : 0 < 25
  t295 : 25 ≤ 25
  h198_0 : MxxRuntime.sliceMatrix (generatedRoot.w_110_0 ctx) 0 1 0 25 (generatedRoot.w_198_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_9  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_9 (generatedRoot.w_110_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_169_1 ctx) (generatedRoot.w_169_2 ctx) (generatedRoot.w_169_3 ctx) (generatedRoot.w_169_4 ctx) (generatedRoot.w_169_5 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) outputs) :
    generatedRoot.Facts_9 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

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
  c9 : generatedRoot.Facts_9 ctx outputs

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
  have r7 := generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) t7.1
  have t8 := t7.2
  have r8 := generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) t8.1
  have t9 := t8.2
  have r9 := generatedRoot.facts_9 (ctx := ctx) (outputs := outputs) t9
  ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9⟩

theorem generatedRoot.h144_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_144.Witness, parallel_generatedRoot_144.Facts (⟨((generatedRoot.w_143_0 ctx), ((generatedRoot.w_143_1 ctx), ((generatedRoot.w_143_2 ctx), ((generatedRoot.w_143_3 ctx), ((generatedRoot.w_143_4 ctx), ((generatedRoot.w_143_5 ctx), ())))))), w⟩ : parallel_generatedRoot_144.Ctx backend hashModel params i) (((generatedRoot.w_144_0 ctx) i), (((generatedRoot.w_144_1 ctx) i), (((generatedRoot.w_144_2 ctx) i), (((generatedRoot.w_144_3 ctx) i), (((generatedRoot.w_144_4 ctx) i), (((generatedRoot.w_144_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h144_5 i
  ⟨w, parallel_generatedRoot_144.facts _ h⟩

theorem generatedRoot.h148_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_148.Witness, parallel_generatedRoot_148.Facts (⟨((generatedRoot.w_147_0 ctx), ((generatedRoot.w_147_1 ctx), ((generatedRoot.w_147_2 ctx), ((generatedRoot.w_147_3 ctx), ((generatedRoot.w_147_4 ctx), ((generatedRoot.w_147_5 ctx), ())))))), w⟩ : parallel_generatedRoot_148.Ctx backend hashModel params i) (((generatedRoot.w_148_0 ctx) i), (((generatedRoot.w_148_1 ctx) i), (((generatedRoot.w_148_2 ctx) i), (((generatedRoot.w_148_3 ctx) i), (((generatedRoot.w_148_4 ctx) i), (((generatedRoot.w_148_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h148_5 i
  ⟨w, parallel_generatedRoot_148.facts _ h⟩

theorem generatedRoot.h149_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_149.Witness, parallel_generatedRoot_149.Facts (⟨((generatedRoot.w_127_0 ctx), ((generatedRoot.w_128_0 ctx), ((generatedRoot.w_129_0 ctx), ((generatedRoot.w_130_0 ctx), ((generatedRoot.w_131_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_132_0 ctx), (((generatedRoot.w_148_4 ctx) i), (((generatedRoot.w_148_0 ctx) i), (((generatedRoot.w_148_2 ctx) i), (((generatedRoot.w_148_5 ctx) i), (((generatedRoot.w_148_1 ctx) i), (((generatedRoot.w_148_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_149.Ctx backend hashModel params i) (((generatedRoot.w_149_0 ctx) i), (((generatedRoot.w_149_1 ctx) i), (((generatedRoot.w_149_2 ctx) i), (((generatedRoot.w_149_3 ctx) i), (((generatedRoot.w_149_4 ctx) i), (((generatedRoot.w_149_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h149_5 i
  ⟨w, parallel_generatedRoot_149.facts _ h⟩

end

end W_Stage_keys_online_1_1

end
