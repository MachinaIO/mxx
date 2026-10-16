import Stage_online_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_online_1

local notation "mxxNat0" => 76731521515431686138455054332509004630949788422460187661496907667850637958884
local notation "mxxNat1" => 8847072971945541729266891833991090237858273327536438370930983040524077886174
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 270
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 65536 3 270
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 270
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 65536 3 540
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 65536 540 270
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 82620
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 306
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_183.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_183.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_183Ctx" => (parallel_generatedRoot_183.Ctx backend hashModel params i_0)

def parallel_generatedRoot_183.w_0_0  (ctx : parallel_generatedRoot_183Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_183.w_1_0  (ctx : parallel_generatedRoot_183Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_183.w_2_0  (ctx : parallel_generatedRoot_183Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_183.w_3_0  (ctx : parallel_generatedRoot_183Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_183.w_4_0  (ctx : parallel_generatedRoot_183Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_183.w_5_0  (ctx : parallel_generatedRoot_183Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_183.w_6_0  (ctx : parallel_generatedRoot_183Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_183.w_6_1  (ctx : parallel_generatedRoot_183Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_183.w_6_2  (ctx : parallel_generatedRoot_183Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_183.w_6_3  (ctx : parallel_generatedRoot_183Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_183.w_6_4  (ctx : parallel_generatedRoot_183Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_183.w_6_5  (ctx : parallel_generatedRoot_183Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_183.w_7_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_183.w_8_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_183.w_9_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_183.w_7_0 ctx) (parallel_generatedRoot_183.w_8_0 ctx)

def parallel_generatedRoot_183.w_10_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_183.w_11_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_183.w_12_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_183.w_10_0 ctx) (parallel_generatedRoot_183.w_11_0 ctx)

def parallel_generatedRoot_183.w_13_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_183.w_14_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_183.w_15_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_183.w_13_0 ctx) (parallel_generatedRoot_183.w_14_0 ctx)

def parallel_generatedRoot_183.w_16_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_183.w_17_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_183.w_18_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_183.w_16_0 ctx) (parallel_generatedRoot_183.w_17_0 ctx)

def parallel_generatedRoot_183.w_19_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_183.w_20_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_183.w_21_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_183.w_19_0 ctx) (parallel_generatedRoot_183.w_20_0 ctx)

def parallel_generatedRoot_183.w_22_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_183.w_23_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_183.w_24_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_183.w_22_0 ctx) (parallel_generatedRoot_183.w_23_0 ctx)

def parallel_generatedRoot_183.w_25_0  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_183.w_25_1  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_183.w_25_2  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_183.w_25_3  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_183.w_25_4  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_183.w_25_5  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_183.outs  (ctx : parallel_generatedRoot_183Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_183.w_25_0 ctx), ((parallel_generatedRoot_183.w_25_1 ctx), ((parallel_generatedRoot_183.w_25_2 ctx), ((parallel_generatedRoot_183.w_25_3 ctx), ((parallel_generatedRoot_183.w_25_4 ctx), ((parallel_generatedRoot_183.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_183.Facts_0  (ctx : parallel_generatedRoot_183Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_183_6 backend hashModel params i_0 i ((parallel_generatedRoot_183.w_0_0 ctx), ((parallel_generatedRoot_183.w_1_0 ctx), ((parallel_generatedRoot_183.w_2_0 ctx), ((parallel_generatedRoot_183.w_3_0 ctx), ((parallel_generatedRoot_183.w_4_0 ctx), ((parallel_generatedRoot_183.w_5_0 ctx), ())))))) (((parallel_generatedRoot_183.w_6_0 ctx) i), (((parallel_generatedRoot_183.w_6_1 ctx) i), (((parallel_generatedRoot_183.w_6_2 ctx) i), (((parallel_generatedRoot_183.w_6_3 ctx) i), (((parallel_generatedRoot_183.w_6_4 ctx) i), (((parallel_generatedRoot_183.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_0 ctx) (0) (parallel_generatedRoot_183.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_0 ctx) (1) (parallel_generatedRoot_183.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_1 ctx) (0) (parallel_generatedRoot_183.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_1 ctx) (1) (parallel_generatedRoot_183.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_2 ctx) (0) (parallel_generatedRoot_183.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_2 ctx) (1) (parallel_generatedRoot_183.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_3 ctx) (0) (parallel_generatedRoot_183.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_3 ctx) (1) (parallel_generatedRoot_183.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_4 ctx) (0) (parallel_generatedRoot_183.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_4 ctx) (1) (parallel_generatedRoot_183.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_5 ctx) (0) (parallel_generatedRoot_183.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_183.w_6_5 ctx) (1) (parallel_generatedRoot_183.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_183_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_183.w_6_0 ctx), ((parallel_generatedRoot_183.w_6_1 ctx), ((parallel_generatedRoot_183.w_6_2 ctx), ((parallel_generatedRoot_183.w_6_3 ctx), ((parallel_generatedRoot_183.w_6_4 ctx), ((parallel_generatedRoot_183.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_183.w_9_0 ctx), ((parallel_generatedRoot_183.w_12_0 ctx), ((parallel_generatedRoot_183.w_15_0 ctx), ((parallel_generatedRoot_183.w_18_0 ctx), ((parallel_generatedRoot_183.w_21_0 ctx), ((parallel_generatedRoot_183.w_24_0 ctx), ())))))) ((parallel_generatedRoot_183.w_25_0 ctx), ((parallel_generatedRoot_183.w_25_1 ctx), ((parallel_generatedRoot_183.w_25_2 ctx), ((parallel_generatedRoot_183.w_25_3 ctx), ((parallel_generatedRoot_183.w_25_4 ctx), ((parallel_generatedRoot_183.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_183.outs ctx

theorem parallel_generatedRoot_183.facts_0  {ctx : parallel_generatedRoot_183Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_183.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_183.w_0_0 ctx) (parallel_generatedRoot_183.w_1_0 ctx) (parallel_generatedRoot_183.w_2_0 ctx) (parallel_generatedRoot_183.w_3_0 ctx) (parallel_generatedRoot_183.w_4_0 ctx) (parallel_generatedRoot_183.w_5_0 ctx) (parallel_generatedRoot_183.w_6_0 ctx) (parallel_generatedRoot_183.w_6_1 ctx) (parallel_generatedRoot_183.w_6_2 ctx) (parallel_generatedRoot_183.w_6_3 ctx) (parallel_generatedRoot_183.w_6_4 ctx) (parallel_generatedRoot_183.w_6_5 ctx) (parallel_generatedRoot_183.w_7_0 ctx) (parallel_generatedRoot_183.w_8_0 ctx) (parallel_generatedRoot_183.w_9_0 ctx) (parallel_generatedRoot_183.w_10_0 ctx) (parallel_generatedRoot_183.w_11_0 ctx) (parallel_generatedRoot_183.w_12_0 ctx) (parallel_generatedRoot_183.w_13_0 ctx) (parallel_generatedRoot_183.w_14_0 ctx) (parallel_generatedRoot_183.w_15_0 ctx) (parallel_generatedRoot_183.w_16_0 ctx) (parallel_generatedRoot_183.w_17_0 ctx) (parallel_generatedRoot_183.w_18_0 ctx) (parallel_generatedRoot_183.w_19_0 ctx) (parallel_generatedRoot_183.w_20_0 ctx) (parallel_generatedRoot_183.w_21_0 ctx) (parallel_generatedRoot_183.w_22_0 ctx) (parallel_generatedRoot_183.w_23_0 ctx) (parallel_generatedRoot_183.w_24_0 ctx) (parallel_generatedRoot_183.w_25_0 ctx) (parallel_generatedRoot_183.w_25_1 ctx) (parallel_generatedRoot_183.w_25_2 ctx) (parallel_generatedRoot_183.w_25_3 ctx) (parallel_generatedRoot_183.w_25_4 ctx) (parallel_generatedRoot_183.w_25_5 ctx) outputs) :
    parallel_generatedRoot_183.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_183.Facts  (ctx : parallel_generatedRoot_183Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_183.Facts_0 ctx outputs

theorem parallel_generatedRoot_183.facts  (ctx : parallel_generatedRoot_183Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_183.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_183.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_183.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_187.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_187.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_187Ctx" => (parallel_generatedRoot_187.Ctx backend hashModel params i_0)

def parallel_generatedRoot_187.w_0_0  (ctx : parallel_generatedRoot_187Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_187.w_1_0  (ctx : parallel_generatedRoot_187Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_187.w_2_0  (ctx : parallel_generatedRoot_187Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_187.w_3_0  (ctx : parallel_generatedRoot_187Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_187.w_4_0  (ctx : parallel_generatedRoot_187Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_187.w_5_0  (ctx : parallel_generatedRoot_187Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_187.w_6_0  (ctx : parallel_generatedRoot_187Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_187.w_6_1  (ctx : parallel_generatedRoot_187Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_187.w_6_2  (ctx : parallel_generatedRoot_187Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_187.w_6_3  (ctx : parallel_generatedRoot_187Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_187.w_6_4  (ctx : parallel_generatedRoot_187Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_187.w_6_5  (ctx : parallel_generatedRoot_187Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_187.w_7_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_187.w_8_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_187.w_9_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_187.w_7_0 ctx) (parallel_generatedRoot_187.w_8_0 ctx)

def parallel_generatedRoot_187.w_10_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_187.w_11_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_187.w_12_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_187.w_10_0 ctx) (parallel_generatedRoot_187.w_11_0 ctx)

def parallel_generatedRoot_187.w_13_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_187.w_14_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_187.w_15_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_187.w_13_0 ctx) (parallel_generatedRoot_187.w_14_0 ctx)

def parallel_generatedRoot_187.w_16_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_187.w_17_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_187.w_18_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_187.w_16_0 ctx) (parallel_generatedRoot_187.w_17_0 ctx)

def parallel_generatedRoot_187.w_19_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_187.w_20_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_187.w_21_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_187.w_19_0 ctx) (parallel_generatedRoot_187.w_20_0 ctx)

def parallel_generatedRoot_187.w_22_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_187.w_23_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_187.w_24_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_187.w_22_0 ctx) (parallel_generatedRoot_187.w_23_0 ctx)

def parallel_generatedRoot_187.w_25_0  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_187.w_25_1  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_187.w_25_2  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_187.w_25_3  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_187.w_25_4  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_187.w_25_5  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_187.outs  (ctx : parallel_generatedRoot_187Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_187.w_25_0 ctx), ((parallel_generatedRoot_187.w_25_1 ctx), ((parallel_generatedRoot_187.w_25_2 ctx), ((parallel_generatedRoot_187.w_25_3 ctx), ((parallel_generatedRoot_187.w_25_4 ctx), ((parallel_generatedRoot_187.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_187.Facts_0  (ctx : parallel_generatedRoot_187Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_187_6 backend hashModel params i_0 i ((parallel_generatedRoot_187.w_0_0 ctx), ((parallel_generatedRoot_187.w_1_0 ctx), ((parallel_generatedRoot_187.w_2_0 ctx), ((parallel_generatedRoot_187.w_3_0 ctx), ((parallel_generatedRoot_187.w_4_0 ctx), ((parallel_generatedRoot_187.w_5_0 ctx), ())))))) (((parallel_generatedRoot_187.w_6_0 ctx) i), (((parallel_generatedRoot_187.w_6_1 ctx) i), (((parallel_generatedRoot_187.w_6_2 ctx) i), (((parallel_generatedRoot_187.w_6_3 ctx) i), (((parallel_generatedRoot_187.w_6_4 ctx) i), (((parallel_generatedRoot_187.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_0 ctx) (0) (parallel_generatedRoot_187.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_0 ctx) (1) (parallel_generatedRoot_187.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_1 ctx) (0) (parallel_generatedRoot_187.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_1 ctx) (1) (parallel_generatedRoot_187.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_2 ctx) (0) (parallel_generatedRoot_187.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_2 ctx) (1) (parallel_generatedRoot_187.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_3 ctx) (0) (parallel_generatedRoot_187.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_3 ctx) (1) (parallel_generatedRoot_187.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_4 ctx) (0) (parallel_generatedRoot_187.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_4 ctx) (1) (parallel_generatedRoot_187.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_5 ctx) (0) (parallel_generatedRoot_187.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_187.w_6_5 ctx) (1) (parallel_generatedRoot_187.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_187_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_187.w_6_0 ctx), ((parallel_generatedRoot_187.w_6_1 ctx), ((parallel_generatedRoot_187.w_6_2 ctx), ((parallel_generatedRoot_187.w_6_3 ctx), ((parallel_generatedRoot_187.w_6_4 ctx), ((parallel_generatedRoot_187.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_187.w_9_0 ctx), ((parallel_generatedRoot_187.w_12_0 ctx), ((parallel_generatedRoot_187.w_15_0 ctx), ((parallel_generatedRoot_187.w_18_0 ctx), ((parallel_generatedRoot_187.w_21_0 ctx), ((parallel_generatedRoot_187.w_24_0 ctx), ())))))) ((parallel_generatedRoot_187.w_25_0 ctx), ((parallel_generatedRoot_187.w_25_1 ctx), ((parallel_generatedRoot_187.w_25_2 ctx), ((parallel_generatedRoot_187.w_25_3 ctx), ((parallel_generatedRoot_187.w_25_4 ctx), ((parallel_generatedRoot_187.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_187.outs ctx

theorem parallel_generatedRoot_187.facts_0  {ctx : parallel_generatedRoot_187Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_187.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_187.w_0_0 ctx) (parallel_generatedRoot_187.w_1_0 ctx) (parallel_generatedRoot_187.w_2_0 ctx) (parallel_generatedRoot_187.w_3_0 ctx) (parallel_generatedRoot_187.w_4_0 ctx) (parallel_generatedRoot_187.w_5_0 ctx) (parallel_generatedRoot_187.w_6_0 ctx) (parallel_generatedRoot_187.w_6_1 ctx) (parallel_generatedRoot_187.w_6_2 ctx) (parallel_generatedRoot_187.w_6_3 ctx) (parallel_generatedRoot_187.w_6_4 ctx) (parallel_generatedRoot_187.w_6_5 ctx) (parallel_generatedRoot_187.w_7_0 ctx) (parallel_generatedRoot_187.w_8_0 ctx) (parallel_generatedRoot_187.w_9_0 ctx) (parallel_generatedRoot_187.w_10_0 ctx) (parallel_generatedRoot_187.w_11_0 ctx) (parallel_generatedRoot_187.w_12_0 ctx) (parallel_generatedRoot_187.w_13_0 ctx) (parallel_generatedRoot_187.w_14_0 ctx) (parallel_generatedRoot_187.w_15_0 ctx) (parallel_generatedRoot_187.w_16_0 ctx) (parallel_generatedRoot_187.w_17_0 ctx) (parallel_generatedRoot_187.w_18_0 ctx) (parallel_generatedRoot_187.w_19_0 ctx) (parallel_generatedRoot_187.w_20_0 ctx) (parallel_generatedRoot_187.w_21_0 ctx) (parallel_generatedRoot_187.w_22_0 ctx) (parallel_generatedRoot_187.w_23_0 ctx) (parallel_generatedRoot_187.w_24_0 ctx) (parallel_generatedRoot_187.w_25_0 ctx) (parallel_generatedRoot_187.w_25_1 ctx) (parallel_generatedRoot_187.w_25_2 ctx) (parallel_generatedRoot_187.w_25_3 ctx) (parallel_generatedRoot_187.w_25_4 ctx) (parallel_generatedRoot_187.w_25_5 ctx) outputs) :
    parallel_generatedRoot_187.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_187.Facts  (ctx : parallel_generatedRoot_187Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_187.Facts_0 ctx outputs

theorem parallel_generatedRoot_187.facts  (ctx : parallel_generatedRoot_187Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_187.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_187.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_187.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_191.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_191.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_191Ctx" => (parallel_generatedRoot_191.Ctx backend hashModel params i_0)

def parallel_generatedRoot_191.w_0_0  (ctx : parallel_generatedRoot_191Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_191.w_1_0  (ctx : parallel_generatedRoot_191Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_191.w_2_0  (ctx : parallel_generatedRoot_191Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_191.w_3_0  (ctx : parallel_generatedRoot_191Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_191.w_4_0  (ctx : parallel_generatedRoot_191Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_191.w_5_0  (ctx : parallel_generatedRoot_191Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_191.w_6_0  (ctx : parallel_generatedRoot_191Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_191.w_6_1  (ctx : parallel_generatedRoot_191Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_191.w_6_2  (ctx : parallel_generatedRoot_191Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_191.w_6_3  (ctx : parallel_generatedRoot_191Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_191.w_6_4  (ctx : parallel_generatedRoot_191Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_191.w_6_5  (ctx : parallel_generatedRoot_191Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_191.w_7_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_191.w_8_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_191.w_9_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_191.w_7_0 ctx) (parallel_generatedRoot_191.w_8_0 ctx)

def parallel_generatedRoot_191.w_10_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_191.w_11_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_191.w_12_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_191.w_10_0 ctx) (parallel_generatedRoot_191.w_11_0 ctx)

def parallel_generatedRoot_191.w_13_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_191.w_14_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_191.w_15_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_191.w_13_0 ctx) (parallel_generatedRoot_191.w_14_0 ctx)

def parallel_generatedRoot_191.w_16_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_191.w_17_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_191.w_18_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_191.w_16_0 ctx) (parallel_generatedRoot_191.w_17_0 ctx)

def parallel_generatedRoot_191.w_19_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_191.w_20_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_191.w_21_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_191.w_19_0 ctx) (parallel_generatedRoot_191.w_20_0 ctx)

def parallel_generatedRoot_191.w_22_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_191.w_23_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_191.w_24_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_191.w_22_0 ctx) (parallel_generatedRoot_191.w_23_0 ctx)

def parallel_generatedRoot_191.w_25_0  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_191.w_25_1  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_191.w_25_2  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_191.w_25_3  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_191.w_25_4  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_191.w_25_5  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_191.outs  (ctx : parallel_generatedRoot_191Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_191.w_25_0 ctx), ((parallel_generatedRoot_191.w_25_1 ctx), ((parallel_generatedRoot_191.w_25_2 ctx), ((parallel_generatedRoot_191.w_25_3 ctx), ((parallel_generatedRoot_191.w_25_4 ctx), ((parallel_generatedRoot_191.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_191.Facts_0  (ctx : parallel_generatedRoot_191Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_191_6 backend hashModel params i_0 i ((parallel_generatedRoot_191.w_0_0 ctx), ((parallel_generatedRoot_191.w_1_0 ctx), ((parallel_generatedRoot_191.w_2_0 ctx), ((parallel_generatedRoot_191.w_3_0 ctx), ((parallel_generatedRoot_191.w_4_0 ctx), ((parallel_generatedRoot_191.w_5_0 ctx), ())))))) (((parallel_generatedRoot_191.w_6_0 ctx) i), (((parallel_generatedRoot_191.w_6_1 ctx) i), (((parallel_generatedRoot_191.w_6_2 ctx) i), (((parallel_generatedRoot_191.w_6_3 ctx) i), (((parallel_generatedRoot_191.w_6_4 ctx) i), (((parallel_generatedRoot_191.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_0 ctx) (0) (parallel_generatedRoot_191.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_0 ctx) (1) (parallel_generatedRoot_191.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_1 ctx) (0) (parallel_generatedRoot_191.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_1 ctx) (1) (parallel_generatedRoot_191.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_2 ctx) (0) (parallel_generatedRoot_191.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_2 ctx) (1) (parallel_generatedRoot_191.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_3 ctx) (0) (parallel_generatedRoot_191.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_3 ctx) (1) (parallel_generatedRoot_191.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_4 ctx) (0) (parallel_generatedRoot_191.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_4 ctx) (1) (parallel_generatedRoot_191.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_5 ctx) (0) (parallel_generatedRoot_191.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_191.w_6_5 ctx) (1) (parallel_generatedRoot_191.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_191_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_191.w_6_0 ctx), ((parallel_generatedRoot_191.w_6_1 ctx), ((parallel_generatedRoot_191.w_6_2 ctx), ((parallel_generatedRoot_191.w_6_3 ctx), ((parallel_generatedRoot_191.w_6_4 ctx), ((parallel_generatedRoot_191.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_191.w_9_0 ctx), ((parallel_generatedRoot_191.w_12_0 ctx), ((parallel_generatedRoot_191.w_15_0 ctx), ((parallel_generatedRoot_191.w_18_0 ctx), ((parallel_generatedRoot_191.w_21_0 ctx), ((parallel_generatedRoot_191.w_24_0 ctx), ())))))) ((parallel_generatedRoot_191.w_25_0 ctx), ((parallel_generatedRoot_191.w_25_1 ctx), ((parallel_generatedRoot_191.w_25_2 ctx), ((parallel_generatedRoot_191.w_25_3 ctx), ((parallel_generatedRoot_191.w_25_4 ctx), ((parallel_generatedRoot_191.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_191.outs ctx

theorem parallel_generatedRoot_191.facts_0  {ctx : parallel_generatedRoot_191Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_191.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_191.w_0_0 ctx) (parallel_generatedRoot_191.w_1_0 ctx) (parallel_generatedRoot_191.w_2_0 ctx) (parallel_generatedRoot_191.w_3_0 ctx) (parallel_generatedRoot_191.w_4_0 ctx) (parallel_generatedRoot_191.w_5_0 ctx) (parallel_generatedRoot_191.w_6_0 ctx) (parallel_generatedRoot_191.w_6_1 ctx) (parallel_generatedRoot_191.w_6_2 ctx) (parallel_generatedRoot_191.w_6_3 ctx) (parallel_generatedRoot_191.w_6_4 ctx) (parallel_generatedRoot_191.w_6_5 ctx) (parallel_generatedRoot_191.w_7_0 ctx) (parallel_generatedRoot_191.w_8_0 ctx) (parallel_generatedRoot_191.w_9_0 ctx) (parallel_generatedRoot_191.w_10_0 ctx) (parallel_generatedRoot_191.w_11_0 ctx) (parallel_generatedRoot_191.w_12_0 ctx) (parallel_generatedRoot_191.w_13_0 ctx) (parallel_generatedRoot_191.w_14_0 ctx) (parallel_generatedRoot_191.w_15_0 ctx) (parallel_generatedRoot_191.w_16_0 ctx) (parallel_generatedRoot_191.w_17_0 ctx) (parallel_generatedRoot_191.w_18_0 ctx) (parallel_generatedRoot_191.w_19_0 ctx) (parallel_generatedRoot_191.w_20_0 ctx) (parallel_generatedRoot_191.w_21_0 ctx) (parallel_generatedRoot_191.w_22_0 ctx) (parallel_generatedRoot_191.w_23_0 ctx) (parallel_generatedRoot_191.w_24_0 ctx) (parallel_generatedRoot_191.w_25_0 ctx) (parallel_generatedRoot_191.w_25_1 ctx) (parallel_generatedRoot_191.w_25_2 ctx) (parallel_generatedRoot_191.w_25_3 ctx) (parallel_generatedRoot_191.w_25_4 ctx) (parallel_generatedRoot_191.w_25_5 ctx) outputs) :
    parallel_generatedRoot_191.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_191.Facts  (ctx : parallel_generatedRoot_191Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_191.Facts_0 ctx outputs

theorem parallel_generatedRoot_191.facts  (ctx : parallel_generatedRoot_191Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_191.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_191.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_191.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_195.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_195.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_195Ctx" => (parallel_generatedRoot_195.Ctx backend hashModel params i_0)

def parallel_generatedRoot_195.w_0_0  (ctx : parallel_generatedRoot_195Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_195.w_1_0  (ctx : parallel_generatedRoot_195Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_195.w_2_0  (ctx : parallel_generatedRoot_195Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_195.w_3_0  (ctx : parallel_generatedRoot_195Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_195.w_4_0  (ctx : parallel_generatedRoot_195Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_195.w_5_0  (ctx : parallel_generatedRoot_195Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_195.w_6_0  (ctx : parallel_generatedRoot_195Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_195.w_6_1  (ctx : parallel_generatedRoot_195Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_195.w_6_2  (ctx : parallel_generatedRoot_195Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_195.w_6_3  (ctx : parallel_generatedRoot_195Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_195.w_6_4  (ctx : parallel_generatedRoot_195Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_195.w_6_5  (ctx : parallel_generatedRoot_195Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_195.w_7_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_195.w_8_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_195.w_9_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_195.w_7_0 ctx) (parallel_generatedRoot_195.w_8_0 ctx)

def parallel_generatedRoot_195.w_10_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_195.w_11_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_195.w_12_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_195.w_10_0 ctx) (parallel_generatedRoot_195.w_11_0 ctx)

def parallel_generatedRoot_195.w_13_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_195.w_14_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_195.w_15_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_195.w_13_0 ctx) (parallel_generatedRoot_195.w_14_0 ctx)

def parallel_generatedRoot_195.w_16_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_195.w_17_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_195.w_18_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_195.w_16_0 ctx) (parallel_generatedRoot_195.w_17_0 ctx)

def parallel_generatedRoot_195.w_19_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_195.w_20_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_195.w_21_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_195.w_19_0 ctx) (parallel_generatedRoot_195.w_20_0 ctx)

def parallel_generatedRoot_195.w_22_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_195.w_23_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_195.w_24_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_195.w_22_0 ctx) (parallel_generatedRoot_195.w_23_0 ctx)

def parallel_generatedRoot_195.w_25_0  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_195.w_25_1  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_195.w_25_2  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_195.w_25_3  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_195.w_25_4  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_195.w_25_5  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_195.outs  (ctx : parallel_generatedRoot_195Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_195.w_25_0 ctx), ((parallel_generatedRoot_195.w_25_1 ctx), ((parallel_generatedRoot_195.w_25_2 ctx), ((parallel_generatedRoot_195.w_25_3 ctx), ((parallel_generatedRoot_195.w_25_4 ctx), ((parallel_generatedRoot_195.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_195.Facts_0  (ctx : parallel_generatedRoot_195Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_195_6 backend hashModel params i_0 i ((parallel_generatedRoot_195.w_0_0 ctx), ((parallel_generatedRoot_195.w_1_0 ctx), ((parallel_generatedRoot_195.w_2_0 ctx), ((parallel_generatedRoot_195.w_3_0 ctx), ((parallel_generatedRoot_195.w_4_0 ctx), ((parallel_generatedRoot_195.w_5_0 ctx), ())))))) (((parallel_generatedRoot_195.w_6_0 ctx) i), (((parallel_generatedRoot_195.w_6_1 ctx) i), (((parallel_generatedRoot_195.w_6_2 ctx) i), (((parallel_generatedRoot_195.w_6_3 ctx) i), (((parallel_generatedRoot_195.w_6_4 ctx) i), (((parallel_generatedRoot_195.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_0 ctx) (0) (parallel_generatedRoot_195.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_0 ctx) (1) (parallel_generatedRoot_195.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_1 ctx) (0) (parallel_generatedRoot_195.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_1 ctx) (1) (parallel_generatedRoot_195.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_2 ctx) (0) (parallel_generatedRoot_195.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_2 ctx) (1) (parallel_generatedRoot_195.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_3 ctx) (0) (parallel_generatedRoot_195.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_3 ctx) (1) (parallel_generatedRoot_195.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_4 ctx) (0) (parallel_generatedRoot_195.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_4 ctx) (1) (parallel_generatedRoot_195.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_5 ctx) (0) (parallel_generatedRoot_195.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_195.w_6_5 ctx) (1) (parallel_generatedRoot_195.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_195_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_195.w_6_0 ctx), ((parallel_generatedRoot_195.w_6_1 ctx), ((parallel_generatedRoot_195.w_6_2 ctx), ((parallel_generatedRoot_195.w_6_3 ctx), ((parallel_generatedRoot_195.w_6_4 ctx), ((parallel_generatedRoot_195.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_195.w_9_0 ctx), ((parallel_generatedRoot_195.w_12_0 ctx), ((parallel_generatedRoot_195.w_15_0 ctx), ((parallel_generatedRoot_195.w_18_0 ctx), ((parallel_generatedRoot_195.w_21_0 ctx), ((parallel_generatedRoot_195.w_24_0 ctx), ())))))) ((parallel_generatedRoot_195.w_25_0 ctx), ((parallel_generatedRoot_195.w_25_1 ctx), ((parallel_generatedRoot_195.w_25_2 ctx), ((parallel_generatedRoot_195.w_25_3 ctx), ((parallel_generatedRoot_195.w_25_4 ctx), ((parallel_generatedRoot_195.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_195.outs ctx

theorem parallel_generatedRoot_195.facts_0  {ctx : parallel_generatedRoot_195Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_195.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_195.w_0_0 ctx) (parallel_generatedRoot_195.w_1_0 ctx) (parallel_generatedRoot_195.w_2_0 ctx) (parallel_generatedRoot_195.w_3_0 ctx) (parallel_generatedRoot_195.w_4_0 ctx) (parallel_generatedRoot_195.w_5_0 ctx) (parallel_generatedRoot_195.w_6_0 ctx) (parallel_generatedRoot_195.w_6_1 ctx) (parallel_generatedRoot_195.w_6_2 ctx) (parallel_generatedRoot_195.w_6_3 ctx) (parallel_generatedRoot_195.w_6_4 ctx) (parallel_generatedRoot_195.w_6_5 ctx) (parallel_generatedRoot_195.w_7_0 ctx) (parallel_generatedRoot_195.w_8_0 ctx) (parallel_generatedRoot_195.w_9_0 ctx) (parallel_generatedRoot_195.w_10_0 ctx) (parallel_generatedRoot_195.w_11_0 ctx) (parallel_generatedRoot_195.w_12_0 ctx) (parallel_generatedRoot_195.w_13_0 ctx) (parallel_generatedRoot_195.w_14_0 ctx) (parallel_generatedRoot_195.w_15_0 ctx) (parallel_generatedRoot_195.w_16_0 ctx) (parallel_generatedRoot_195.w_17_0 ctx) (parallel_generatedRoot_195.w_18_0 ctx) (parallel_generatedRoot_195.w_19_0 ctx) (parallel_generatedRoot_195.w_20_0 ctx) (parallel_generatedRoot_195.w_21_0 ctx) (parallel_generatedRoot_195.w_22_0 ctx) (parallel_generatedRoot_195.w_23_0 ctx) (parallel_generatedRoot_195.w_24_0 ctx) (parallel_generatedRoot_195.w_25_0 ctx) (parallel_generatedRoot_195.w_25_1 ctx) (parallel_generatedRoot_195.w_25_2 ctx) (parallel_generatedRoot_195.w_25_3 ctx) (parallel_generatedRoot_195.w_25_4 ctx) (parallel_generatedRoot_195.w_25_5 ctx) outputs) :
    parallel_generatedRoot_195.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_195.Facts  (ctx : parallel_generatedRoot_195Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_195.Facts_0 ctx outputs

theorem parallel_generatedRoot_195.facts  (ctx : parallel_generatedRoot_195Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_195.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_195.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_195.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_199.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_199.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_199Ctx" => (parallel_generatedRoot_199.Ctx backend hashModel params i_0)

def parallel_generatedRoot_199.w_0_0  (ctx : parallel_generatedRoot_199Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_199.w_1_0  (ctx : parallel_generatedRoot_199Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_199.w_2_0  (ctx : parallel_generatedRoot_199Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_199.w_3_0  (ctx : parallel_generatedRoot_199Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_199.w_4_0  (ctx : parallel_generatedRoot_199Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_199.w_5_0  (ctx : parallel_generatedRoot_199Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_199.w_6_0  (ctx : parallel_generatedRoot_199Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_199.w_6_1  (ctx : parallel_generatedRoot_199Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_199.w_6_2  (ctx : parallel_generatedRoot_199Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_199.w_6_3  (ctx : parallel_generatedRoot_199Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_199.w_6_4  (ctx : parallel_generatedRoot_199Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_199.w_6_5  (ctx : parallel_generatedRoot_199Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_199.w_7_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_199.w_8_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_199.w_9_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_199.w_7_0 ctx) (parallel_generatedRoot_199.w_8_0 ctx)

def parallel_generatedRoot_199.w_10_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_199.w_11_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_199.w_12_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_199.w_10_0 ctx) (parallel_generatedRoot_199.w_11_0 ctx)

def parallel_generatedRoot_199.w_13_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_199.w_14_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_199.w_15_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_199.w_13_0 ctx) (parallel_generatedRoot_199.w_14_0 ctx)

def parallel_generatedRoot_199.w_16_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_199.w_17_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_199.w_18_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_199.w_16_0 ctx) (parallel_generatedRoot_199.w_17_0 ctx)

def parallel_generatedRoot_199.w_19_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_199.w_20_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_199.w_21_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_199.w_19_0 ctx) (parallel_generatedRoot_199.w_20_0 ctx)

def parallel_generatedRoot_199.w_22_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_199.w_23_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_199.w_24_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_199.w_22_0 ctx) (parallel_generatedRoot_199.w_23_0 ctx)

def parallel_generatedRoot_199.w_25_0  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_199.w_25_1  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_199.w_25_2  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_199.w_25_3  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_199.w_25_4  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_199.w_25_5  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_199.outs  (ctx : parallel_generatedRoot_199Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_199.w_25_0 ctx), ((parallel_generatedRoot_199.w_25_1 ctx), ((parallel_generatedRoot_199.w_25_2 ctx), ((parallel_generatedRoot_199.w_25_3 ctx), ((parallel_generatedRoot_199.w_25_4 ctx), ((parallel_generatedRoot_199.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_199.Facts_0  (ctx : parallel_generatedRoot_199Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_199_6 backend hashModel params i_0 i ((parallel_generatedRoot_199.w_0_0 ctx), ((parallel_generatedRoot_199.w_1_0 ctx), ((parallel_generatedRoot_199.w_2_0 ctx), ((parallel_generatedRoot_199.w_3_0 ctx), ((parallel_generatedRoot_199.w_4_0 ctx), ((parallel_generatedRoot_199.w_5_0 ctx), ())))))) (((parallel_generatedRoot_199.w_6_0 ctx) i), (((parallel_generatedRoot_199.w_6_1 ctx) i), (((parallel_generatedRoot_199.w_6_2 ctx) i), (((parallel_generatedRoot_199.w_6_3 ctx) i), (((parallel_generatedRoot_199.w_6_4 ctx) i), (((parallel_generatedRoot_199.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_0 ctx) (0) (parallel_generatedRoot_199.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_0 ctx) (1) (parallel_generatedRoot_199.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_1 ctx) (0) (parallel_generatedRoot_199.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_1 ctx) (1) (parallel_generatedRoot_199.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_2 ctx) (0) (parallel_generatedRoot_199.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_2 ctx) (1) (parallel_generatedRoot_199.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_3 ctx) (0) (parallel_generatedRoot_199.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_3 ctx) (1) (parallel_generatedRoot_199.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_4 ctx) (0) (parallel_generatedRoot_199.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_4 ctx) (1) (parallel_generatedRoot_199.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_5 ctx) (0) (parallel_generatedRoot_199.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_199.w_6_5 ctx) (1) (parallel_generatedRoot_199.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_199_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_199.w_6_0 ctx), ((parallel_generatedRoot_199.w_6_1 ctx), ((parallel_generatedRoot_199.w_6_2 ctx), ((parallel_generatedRoot_199.w_6_3 ctx), ((parallel_generatedRoot_199.w_6_4 ctx), ((parallel_generatedRoot_199.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_199.w_9_0 ctx), ((parallel_generatedRoot_199.w_12_0 ctx), ((parallel_generatedRoot_199.w_15_0 ctx), ((parallel_generatedRoot_199.w_18_0 ctx), ((parallel_generatedRoot_199.w_21_0 ctx), ((parallel_generatedRoot_199.w_24_0 ctx), ())))))) ((parallel_generatedRoot_199.w_25_0 ctx), ((parallel_generatedRoot_199.w_25_1 ctx), ((parallel_generatedRoot_199.w_25_2 ctx), ((parallel_generatedRoot_199.w_25_3 ctx), ((parallel_generatedRoot_199.w_25_4 ctx), ((parallel_generatedRoot_199.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_199.outs ctx

theorem parallel_generatedRoot_199.facts_0  {ctx : parallel_generatedRoot_199Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_199.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_199.w_0_0 ctx) (parallel_generatedRoot_199.w_1_0 ctx) (parallel_generatedRoot_199.w_2_0 ctx) (parallel_generatedRoot_199.w_3_0 ctx) (parallel_generatedRoot_199.w_4_0 ctx) (parallel_generatedRoot_199.w_5_0 ctx) (parallel_generatedRoot_199.w_6_0 ctx) (parallel_generatedRoot_199.w_6_1 ctx) (parallel_generatedRoot_199.w_6_2 ctx) (parallel_generatedRoot_199.w_6_3 ctx) (parallel_generatedRoot_199.w_6_4 ctx) (parallel_generatedRoot_199.w_6_5 ctx) (parallel_generatedRoot_199.w_7_0 ctx) (parallel_generatedRoot_199.w_8_0 ctx) (parallel_generatedRoot_199.w_9_0 ctx) (parallel_generatedRoot_199.w_10_0 ctx) (parallel_generatedRoot_199.w_11_0 ctx) (parallel_generatedRoot_199.w_12_0 ctx) (parallel_generatedRoot_199.w_13_0 ctx) (parallel_generatedRoot_199.w_14_0 ctx) (parallel_generatedRoot_199.w_15_0 ctx) (parallel_generatedRoot_199.w_16_0 ctx) (parallel_generatedRoot_199.w_17_0 ctx) (parallel_generatedRoot_199.w_18_0 ctx) (parallel_generatedRoot_199.w_19_0 ctx) (parallel_generatedRoot_199.w_20_0 ctx) (parallel_generatedRoot_199.w_21_0 ctx) (parallel_generatedRoot_199.w_22_0 ctx) (parallel_generatedRoot_199.w_23_0 ctx) (parallel_generatedRoot_199.w_24_0 ctx) (parallel_generatedRoot_199.w_25_0 ctx) (parallel_generatedRoot_199.w_25_1 ctx) (parallel_generatedRoot_199.w_25_2 ctx) (parallel_generatedRoot_199.w_25_3 ctx) (parallel_generatedRoot_199.w_25_4 ctx) (parallel_generatedRoot_199.w_25_5 ctx) outputs) :
    parallel_generatedRoot_199.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_199.Facts  (ctx : parallel_generatedRoot_199Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_199.Facts_0 ctx outputs

theorem parallel_generatedRoot_199.facts  (ctx : parallel_generatedRoot_199Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_199.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_199.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_199.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_203.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_203.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_203Ctx" => (parallel_generatedRoot_203.Ctx backend hashModel params i_0)

def parallel_generatedRoot_203.w_0_0  (ctx : parallel_generatedRoot_203Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_203.w_1_0  (ctx : parallel_generatedRoot_203Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_203.w_2_0  (ctx : parallel_generatedRoot_203Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_203.w_3_0  (ctx : parallel_generatedRoot_203Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_203.w_4_0  (ctx : parallel_generatedRoot_203Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_203.w_5_0  (ctx : parallel_generatedRoot_203Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_203.w_6_0  (ctx : parallel_generatedRoot_203Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_203.w_6_1  (ctx : parallel_generatedRoot_203Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_203.w_6_2  (ctx : parallel_generatedRoot_203Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_203.w_6_3  (ctx : parallel_generatedRoot_203Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_203.w_6_4  (ctx : parallel_generatedRoot_203Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_203.w_6_5  (ctx : parallel_generatedRoot_203Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_203.w_7_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_203.w_8_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_203.w_9_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_203.w_7_0 ctx) (parallel_generatedRoot_203.w_8_0 ctx)

def parallel_generatedRoot_203.w_10_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_203.w_11_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_203.w_12_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_203.w_10_0 ctx) (parallel_generatedRoot_203.w_11_0 ctx)

def parallel_generatedRoot_203.w_13_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_203.w_14_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_203.w_15_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_203.w_13_0 ctx) (parallel_generatedRoot_203.w_14_0 ctx)

def parallel_generatedRoot_203.w_16_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_203.w_17_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_203.w_18_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_203.w_16_0 ctx) (parallel_generatedRoot_203.w_17_0 ctx)

def parallel_generatedRoot_203.w_19_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_203.w_20_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_203.w_21_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_203.w_19_0 ctx) (parallel_generatedRoot_203.w_20_0 ctx)

def parallel_generatedRoot_203.w_22_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_203.w_23_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_203.w_24_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_203.w_22_0 ctx) (parallel_generatedRoot_203.w_23_0 ctx)

def parallel_generatedRoot_203.w_25_0  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_203.w_25_1  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_203.w_25_2  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_203.w_25_3  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_203.w_25_4  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_203.w_25_5  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_203.outs  (ctx : parallel_generatedRoot_203Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_203.w_25_0 ctx), ((parallel_generatedRoot_203.w_25_1 ctx), ((parallel_generatedRoot_203.w_25_2 ctx), ((parallel_generatedRoot_203.w_25_3 ctx), ((parallel_generatedRoot_203.w_25_4 ctx), ((parallel_generatedRoot_203.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_203.Facts_0  (ctx : parallel_generatedRoot_203Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_203_6 backend hashModel params i_0 i ((parallel_generatedRoot_203.w_0_0 ctx), ((parallel_generatedRoot_203.w_1_0 ctx), ((parallel_generatedRoot_203.w_2_0 ctx), ((parallel_generatedRoot_203.w_3_0 ctx), ((parallel_generatedRoot_203.w_4_0 ctx), ((parallel_generatedRoot_203.w_5_0 ctx), ())))))) (((parallel_generatedRoot_203.w_6_0 ctx) i), (((parallel_generatedRoot_203.w_6_1 ctx) i), (((parallel_generatedRoot_203.w_6_2 ctx) i), (((parallel_generatedRoot_203.w_6_3 ctx) i), (((parallel_generatedRoot_203.w_6_4 ctx) i), (((parallel_generatedRoot_203.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_0 ctx) (0) (parallel_generatedRoot_203.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_0 ctx) (1) (parallel_generatedRoot_203.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_1 ctx) (0) (parallel_generatedRoot_203.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_1 ctx) (1) (parallel_generatedRoot_203.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_2 ctx) (0) (parallel_generatedRoot_203.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_2 ctx) (1) (parallel_generatedRoot_203.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_3 ctx) (0) (parallel_generatedRoot_203.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_3 ctx) (1) (parallel_generatedRoot_203.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_4 ctx) (0) (parallel_generatedRoot_203.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_4 ctx) (1) (parallel_generatedRoot_203.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_5 ctx) (0) (parallel_generatedRoot_203.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_203.w_6_5 ctx) (1) (parallel_generatedRoot_203.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_203_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_203.w_6_0 ctx), ((parallel_generatedRoot_203.w_6_1 ctx), ((parallel_generatedRoot_203.w_6_2 ctx), ((parallel_generatedRoot_203.w_6_3 ctx), ((parallel_generatedRoot_203.w_6_4 ctx), ((parallel_generatedRoot_203.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_203.w_9_0 ctx), ((parallel_generatedRoot_203.w_12_0 ctx), ((parallel_generatedRoot_203.w_15_0 ctx), ((parallel_generatedRoot_203.w_18_0 ctx), ((parallel_generatedRoot_203.w_21_0 ctx), ((parallel_generatedRoot_203.w_24_0 ctx), ())))))) ((parallel_generatedRoot_203.w_25_0 ctx), ((parallel_generatedRoot_203.w_25_1 ctx), ((parallel_generatedRoot_203.w_25_2 ctx), ((parallel_generatedRoot_203.w_25_3 ctx), ((parallel_generatedRoot_203.w_25_4 ctx), ((parallel_generatedRoot_203.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_203.outs ctx

theorem parallel_generatedRoot_203.facts_0  {ctx : parallel_generatedRoot_203Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_203.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_203.w_0_0 ctx) (parallel_generatedRoot_203.w_1_0 ctx) (parallel_generatedRoot_203.w_2_0 ctx) (parallel_generatedRoot_203.w_3_0 ctx) (parallel_generatedRoot_203.w_4_0 ctx) (parallel_generatedRoot_203.w_5_0 ctx) (parallel_generatedRoot_203.w_6_0 ctx) (parallel_generatedRoot_203.w_6_1 ctx) (parallel_generatedRoot_203.w_6_2 ctx) (parallel_generatedRoot_203.w_6_3 ctx) (parallel_generatedRoot_203.w_6_4 ctx) (parallel_generatedRoot_203.w_6_5 ctx) (parallel_generatedRoot_203.w_7_0 ctx) (parallel_generatedRoot_203.w_8_0 ctx) (parallel_generatedRoot_203.w_9_0 ctx) (parallel_generatedRoot_203.w_10_0 ctx) (parallel_generatedRoot_203.w_11_0 ctx) (parallel_generatedRoot_203.w_12_0 ctx) (parallel_generatedRoot_203.w_13_0 ctx) (parallel_generatedRoot_203.w_14_0 ctx) (parallel_generatedRoot_203.w_15_0 ctx) (parallel_generatedRoot_203.w_16_0 ctx) (parallel_generatedRoot_203.w_17_0 ctx) (parallel_generatedRoot_203.w_18_0 ctx) (parallel_generatedRoot_203.w_19_0 ctx) (parallel_generatedRoot_203.w_20_0 ctx) (parallel_generatedRoot_203.w_21_0 ctx) (parallel_generatedRoot_203.w_22_0 ctx) (parallel_generatedRoot_203.w_23_0 ctx) (parallel_generatedRoot_203.w_24_0 ctx) (parallel_generatedRoot_203.w_25_0 ctx) (parallel_generatedRoot_203.w_25_1 ctx) (parallel_generatedRoot_203.w_25_2 ctx) (parallel_generatedRoot_203.w_25_3 ctx) (parallel_generatedRoot_203.w_25_4 ctx) (parallel_generatedRoot_203.w_25_5 ctx) outputs) :
    parallel_generatedRoot_203.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_203.Facts  (ctx : parallel_generatedRoot_203Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_203.Facts_0 ctx outputs

theorem parallel_generatedRoot_203.facts  (ctx : parallel_generatedRoot_203Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_203.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_203.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_203.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_207.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_207.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_207Ctx" => (parallel_generatedRoot_207.Ctx backend hashModel params i_0)

def parallel_generatedRoot_207.w_0_0  (ctx : parallel_generatedRoot_207Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_207.w_1_0  (ctx : parallel_generatedRoot_207Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_207.w_2_0  (ctx : parallel_generatedRoot_207Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_207.w_3_0  (ctx : parallel_generatedRoot_207Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_207.w_4_0  (ctx : parallel_generatedRoot_207Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_207.w_5_0  (ctx : parallel_generatedRoot_207Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_207.w_6_0  (ctx : parallel_generatedRoot_207Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_207.w_6_1  (ctx : parallel_generatedRoot_207Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_207.w_6_2  (ctx : parallel_generatedRoot_207Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_207.w_6_3  (ctx : parallel_generatedRoot_207Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_207.w_6_4  (ctx : parallel_generatedRoot_207Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_207.w_6_5  (ctx : parallel_generatedRoot_207Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_207.w_7_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_207.w_8_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_207.w_9_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_207.w_7_0 ctx) (parallel_generatedRoot_207.w_8_0 ctx)

def parallel_generatedRoot_207.w_10_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_207.w_11_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_207.w_12_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_207.w_10_0 ctx) (parallel_generatedRoot_207.w_11_0 ctx)

def parallel_generatedRoot_207.w_13_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_207.w_14_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_207.w_15_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_207.w_13_0 ctx) (parallel_generatedRoot_207.w_14_0 ctx)

def parallel_generatedRoot_207.w_16_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_207.w_17_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_207.w_18_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_207.w_16_0 ctx) (parallel_generatedRoot_207.w_17_0 ctx)

def parallel_generatedRoot_207.w_19_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_207.w_20_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_207.w_21_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_207.w_19_0 ctx) (parallel_generatedRoot_207.w_20_0 ctx)

def parallel_generatedRoot_207.w_22_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_207.w_23_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_207.w_24_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_207.w_22_0 ctx) (parallel_generatedRoot_207.w_23_0 ctx)

def parallel_generatedRoot_207.w_25_0  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_207.w_25_1  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_207.w_25_2  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_207.w_25_3  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_207.w_25_4  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_207.w_25_5  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_207.outs  (ctx : parallel_generatedRoot_207Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_207.w_25_0 ctx), ((parallel_generatedRoot_207.w_25_1 ctx), ((parallel_generatedRoot_207.w_25_2 ctx), ((parallel_generatedRoot_207.w_25_3 ctx), ((parallel_generatedRoot_207.w_25_4 ctx), ((parallel_generatedRoot_207.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_207.Facts_0  (ctx : parallel_generatedRoot_207Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_207_6 backend hashModel params i_0 i ((parallel_generatedRoot_207.w_0_0 ctx), ((parallel_generatedRoot_207.w_1_0 ctx), ((parallel_generatedRoot_207.w_2_0 ctx), ((parallel_generatedRoot_207.w_3_0 ctx), ((parallel_generatedRoot_207.w_4_0 ctx), ((parallel_generatedRoot_207.w_5_0 ctx), ())))))) (((parallel_generatedRoot_207.w_6_0 ctx) i), (((parallel_generatedRoot_207.w_6_1 ctx) i), (((parallel_generatedRoot_207.w_6_2 ctx) i), (((parallel_generatedRoot_207.w_6_3 ctx) i), (((parallel_generatedRoot_207.w_6_4 ctx) i), (((parallel_generatedRoot_207.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_0 ctx) (0) (parallel_generatedRoot_207.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_0 ctx) (1) (parallel_generatedRoot_207.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_1 ctx) (0) (parallel_generatedRoot_207.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_1 ctx) (1) (parallel_generatedRoot_207.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_2 ctx) (0) (parallel_generatedRoot_207.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_2 ctx) (1) (parallel_generatedRoot_207.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_3 ctx) (0) (parallel_generatedRoot_207.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_3 ctx) (1) (parallel_generatedRoot_207.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_4 ctx) (0) (parallel_generatedRoot_207.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_4 ctx) (1) (parallel_generatedRoot_207.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_5 ctx) (0) (parallel_generatedRoot_207.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_207.w_6_5 ctx) (1) (parallel_generatedRoot_207.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_207_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_207.w_6_0 ctx), ((parallel_generatedRoot_207.w_6_1 ctx), ((parallel_generatedRoot_207.w_6_2 ctx), ((parallel_generatedRoot_207.w_6_3 ctx), ((parallel_generatedRoot_207.w_6_4 ctx), ((parallel_generatedRoot_207.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_207.w_9_0 ctx), ((parallel_generatedRoot_207.w_12_0 ctx), ((parallel_generatedRoot_207.w_15_0 ctx), ((parallel_generatedRoot_207.w_18_0 ctx), ((parallel_generatedRoot_207.w_21_0 ctx), ((parallel_generatedRoot_207.w_24_0 ctx), ())))))) ((parallel_generatedRoot_207.w_25_0 ctx), ((parallel_generatedRoot_207.w_25_1 ctx), ((parallel_generatedRoot_207.w_25_2 ctx), ((parallel_generatedRoot_207.w_25_3 ctx), ((parallel_generatedRoot_207.w_25_4 ctx), ((parallel_generatedRoot_207.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_207.outs ctx

theorem parallel_generatedRoot_207.facts_0  {ctx : parallel_generatedRoot_207Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_207.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_207.w_0_0 ctx) (parallel_generatedRoot_207.w_1_0 ctx) (parallel_generatedRoot_207.w_2_0 ctx) (parallel_generatedRoot_207.w_3_0 ctx) (parallel_generatedRoot_207.w_4_0 ctx) (parallel_generatedRoot_207.w_5_0 ctx) (parallel_generatedRoot_207.w_6_0 ctx) (parallel_generatedRoot_207.w_6_1 ctx) (parallel_generatedRoot_207.w_6_2 ctx) (parallel_generatedRoot_207.w_6_3 ctx) (parallel_generatedRoot_207.w_6_4 ctx) (parallel_generatedRoot_207.w_6_5 ctx) (parallel_generatedRoot_207.w_7_0 ctx) (parallel_generatedRoot_207.w_8_0 ctx) (parallel_generatedRoot_207.w_9_0 ctx) (parallel_generatedRoot_207.w_10_0 ctx) (parallel_generatedRoot_207.w_11_0 ctx) (parallel_generatedRoot_207.w_12_0 ctx) (parallel_generatedRoot_207.w_13_0 ctx) (parallel_generatedRoot_207.w_14_0 ctx) (parallel_generatedRoot_207.w_15_0 ctx) (parallel_generatedRoot_207.w_16_0 ctx) (parallel_generatedRoot_207.w_17_0 ctx) (parallel_generatedRoot_207.w_18_0 ctx) (parallel_generatedRoot_207.w_19_0 ctx) (parallel_generatedRoot_207.w_20_0 ctx) (parallel_generatedRoot_207.w_21_0 ctx) (parallel_generatedRoot_207.w_22_0 ctx) (parallel_generatedRoot_207.w_23_0 ctx) (parallel_generatedRoot_207.w_24_0 ctx) (parallel_generatedRoot_207.w_25_0 ctx) (parallel_generatedRoot_207.w_25_1 ctx) (parallel_generatedRoot_207.w_25_2 ctx) (parallel_generatedRoot_207.w_25_3 ctx) (parallel_generatedRoot_207.w_25_4 ctx) (parallel_generatedRoot_207.w_25_5 ctx) outputs) :
    parallel_generatedRoot_207.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_207.Facts  (ctx : parallel_generatedRoot_207Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_207.Facts_0 ctx outputs

theorem parallel_generatedRoot_207.facts  (ctx : parallel_generatedRoot_207Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_207.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_207.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_207.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_211.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_211.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_211Ctx" => (parallel_generatedRoot_211.Ctx backend hashModel params i_0)

def parallel_generatedRoot_211.w_0_0  (ctx : parallel_generatedRoot_211Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_211.w_1_0  (ctx : parallel_generatedRoot_211Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_211.w_2_0  (ctx : parallel_generatedRoot_211Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_211.w_3_0  (ctx : parallel_generatedRoot_211Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_211.w_4_0  (ctx : parallel_generatedRoot_211Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_211.w_5_0  (ctx : parallel_generatedRoot_211Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_211.w_6_0  (ctx : parallel_generatedRoot_211Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_211.w_6_1  (ctx : parallel_generatedRoot_211Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_211.w_6_2  (ctx : parallel_generatedRoot_211Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_211.w_6_3  (ctx : parallel_generatedRoot_211Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_211.w_6_4  (ctx : parallel_generatedRoot_211Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_211.w_6_5  (ctx : parallel_generatedRoot_211Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_211.w_7_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_211.w_8_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_211.w_9_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_211.w_7_0 ctx) (parallel_generatedRoot_211.w_8_0 ctx)

def parallel_generatedRoot_211.w_10_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_211.w_11_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_211.w_12_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_211.w_10_0 ctx) (parallel_generatedRoot_211.w_11_0 ctx)

def parallel_generatedRoot_211.w_13_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_211.w_14_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_211.w_15_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_211.w_13_0 ctx) (parallel_generatedRoot_211.w_14_0 ctx)

def parallel_generatedRoot_211.w_16_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_211.w_17_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_211.w_18_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_211.w_16_0 ctx) (parallel_generatedRoot_211.w_17_0 ctx)

def parallel_generatedRoot_211.w_19_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_211.w_20_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_211.w_21_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_211.w_19_0 ctx) (parallel_generatedRoot_211.w_20_0 ctx)

def parallel_generatedRoot_211.w_22_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_211.w_23_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_211.w_24_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_211.w_22_0 ctx) (parallel_generatedRoot_211.w_23_0 ctx)

def parallel_generatedRoot_211.w_25_0  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_211.w_25_1  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_211.w_25_2  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_211.w_25_3  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_211.w_25_4  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_211.w_25_5  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_211.outs  (ctx : parallel_generatedRoot_211Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_211.w_25_0 ctx), ((parallel_generatedRoot_211.w_25_1 ctx), ((parallel_generatedRoot_211.w_25_2 ctx), ((parallel_generatedRoot_211.w_25_3 ctx), ((parallel_generatedRoot_211.w_25_4 ctx), ((parallel_generatedRoot_211.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_211.Facts_0  (ctx : parallel_generatedRoot_211Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_211_6 backend hashModel params i_0 i ((parallel_generatedRoot_211.w_0_0 ctx), ((parallel_generatedRoot_211.w_1_0 ctx), ((parallel_generatedRoot_211.w_2_0 ctx), ((parallel_generatedRoot_211.w_3_0 ctx), ((parallel_generatedRoot_211.w_4_0 ctx), ((parallel_generatedRoot_211.w_5_0 ctx), ())))))) (((parallel_generatedRoot_211.w_6_0 ctx) i), (((parallel_generatedRoot_211.w_6_1 ctx) i), (((parallel_generatedRoot_211.w_6_2 ctx) i), (((parallel_generatedRoot_211.w_6_3 ctx) i), (((parallel_generatedRoot_211.w_6_4 ctx) i), (((parallel_generatedRoot_211.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_0 ctx) (0) (parallel_generatedRoot_211.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_0 ctx) (1) (parallel_generatedRoot_211.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_1 ctx) (0) (parallel_generatedRoot_211.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_1 ctx) (1) (parallel_generatedRoot_211.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_2 ctx) (0) (parallel_generatedRoot_211.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_2 ctx) (1) (parallel_generatedRoot_211.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_3 ctx) (0) (parallel_generatedRoot_211.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_3 ctx) (1) (parallel_generatedRoot_211.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_4 ctx) (0) (parallel_generatedRoot_211.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_4 ctx) (1) (parallel_generatedRoot_211.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_5 ctx) (0) (parallel_generatedRoot_211.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_211.w_6_5 ctx) (1) (parallel_generatedRoot_211.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_211_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_211.w_6_0 ctx), ((parallel_generatedRoot_211.w_6_1 ctx), ((parallel_generatedRoot_211.w_6_2 ctx), ((parallel_generatedRoot_211.w_6_3 ctx), ((parallel_generatedRoot_211.w_6_4 ctx), ((parallel_generatedRoot_211.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_211.w_9_0 ctx), ((parallel_generatedRoot_211.w_12_0 ctx), ((parallel_generatedRoot_211.w_15_0 ctx), ((parallel_generatedRoot_211.w_18_0 ctx), ((parallel_generatedRoot_211.w_21_0 ctx), ((parallel_generatedRoot_211.w_24_0 ctx), ())))))) ((parallel_generatedRoot_211.w_25_0 ctx), ((parallel_generatedRoot_211.w_25_1 ctx), ((parallel_generatedRoot_211.w_25_2 ctx), ((parallel_generatedRoot_211.w_25_3 ctx), ((parallel_generatedRoot_211.w_25_4 ctx), ((parallel_generatedRoot_211.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_211.outs ctx

theorem parallel_generatedRoot_211.facts_0  {ctx : parallel_generatedRoot_211Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_211.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_211.w_0_0 ctx) (parallel_generatedRoot_211.w_1_0 ctx) (parallel_generatedRoot_211.w_2_0 ctx) (parallel_generatedRoot_211.w_3_0 ctx) (parallel_generatedRoot_211.w_4_0 ctx) (parallel_generatedRoot_211.w_5_0 ctx) (parallel_generatedRoot_211.w_6_0 ctx) (parallel_generatedRoot_211.w_6_1 ctx) (parallel_generatedRoot_211.w_6_2 ctx) (parallel_generatedRoot_211.w_6_3 ctx) (parallel_generatedRoot_211.w_6_4 ctx) (parallel_generatedRoot_211.w_6_5 ctx) (parallel_generatedRoot_211.w_7_0 ctx) (parallel_generatedRoot_211.w_8_0 ctx) (parallel_generatedRoot_211.w_9_0 ctx) (parallel_generatedRoot_211.w_10_0 ctx) (parallel_generatedRoot_211.w_11_0 ctx) (parallel_generatedRoot_211.w_12_0 ctx) (parallel_generatedRoot_211.w_13_0 ctx) (parallel_generatedRoot_211.w_14_0 ctx) (parallel_generatedRoot_211.w_15_0 ctx) (parallel_generatedRoot_211.w_16_0 ctx) (parallel_generatedRoot_211.w_17_0 ctx) (parallel_generatedRoot_211.w_18_0 ctx) (parallel_generatedRoot_211.w_19_0 ctx) (parallel_generatedRoot_211.w_20_0 ctx) (parallel_generatedRoot_211.w_21_0 ctx) (parallel_generatedRoot_211.w_22_0 ctx) (parallel_generatedRoot_211.w_23_0 ctx) (parallel_generatedRoot_211.w_24_0 ctx) (parallel_generatedRoot_211.w_25_0 ctx) (parallel_generatedRoot_211.w_25_1 ctx) (parallel_generatedRoot_211.w_25_2 ctx) (parallel_generatedRoot_211.w_25_3 ctx) (parallel_generatedRoot_211.w_25_4 ctx) (parallel_generatedRoot_211.w_25_5 ctx) outputs) :
    parallel_generatedRoot_211.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_211.Facts  (ctx : parallel_generatedRoot_211Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_211.Facts_0 ctx outputs

theorem parallel_generatedRoot_211.facts  (ctx : parallel_generatedRoot_211Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_211.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_211.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_211.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_215.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_215.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_215Ctx" => (parallel_generatedRoot_215.Ctx backend hashModel params i_0)

def parallel_generatedRoot_215.w_0_0  (ctx : parallel_generatedRoot_215Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_215.w_1_0  (ctx : parallel_generatedRoot_215Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_215.w_2_0  (ctx : parallel_generatedRoot_215Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_215.w_3_0  (ctx : parallel_generatedRoot_215Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_215.w_4_0  (ctx : parallel_generatedRoot_215Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_215.w_5_0  (ctx : parallel_generatedRoot_215Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_215.w_6_0  (ctx : parallel_generatedRoot_215Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_215.w_6_1  (ctx : parallel_generatedRoot_215Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_215.w_6_2  (ctx : parallel_generatedRoot_215Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_215.w_6_3  (ctx : parallel_generatedRoot_215Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_215.w_6_4  (ctx : parallel_generatedRoot_215Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_215.w_6_5  (ctx : parallel_generatedRoot_215Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_215.w_7_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_215.w_8_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_215.w_9_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_215.w_7_0 ctx) (parallel_generatedRoot_215.w_8_0 ctx)

def parallel_generatedRoot_215.w_10_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_215.w_11_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_215.w_12_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_215.w_10_0 ctx) (parallel_generatedRoot_215.w_11_0 ctx)

def parallel_generatedRoot_215.w_13_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_215.w_14_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_215.w_15_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_215.w_13_0 ctx) (parallel_generatedRoot_215.w_14_0 ctx)

def parallel_generatedRoot_215.w_16_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_215.w_17_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_215.w_18_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_215.w_16_0 ctx) (parallel_generatedRoot_215.w_17_0 ctx)

def parallel_generatedRoot_215.w_19_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_215.w_20_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_215.w_21_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_215.w_19_0 ctx) (parallel_generatedRoot_215.w_20_0 ctx)

def parallel_generatedRoot_215.w_22_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_215.w_23_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_215.w_24_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_215.w_22_0 ctx) (parallel_generatedRoot_215.w_23_0 ctx)

def parallel_generatedRoot_215.w_25_0  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_215.w_25_1  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_215.w_25_2  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_215.w_25_3  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_215.w_25_4  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_215.w_25_5  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_215.outs  (ctx : parallel_generatedRoot_215Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_215.w_25_0 ctx), ((parallel_generatedRoot_215.w_25_1 ctx), ((parallel_generatedRoot_215.w_25_2 ctx), ((parallel_generatedRoot_215.w_25_3 ctx), ((parallel_generatedRoot_215.w_25_4 ctx), ((parallel_generatedRoot_215.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_215.Facts_0  (ctx : parallel_generatedRoot_215Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_215_6 backend hashModel params i_0 i ((parallel_generatedRoot_215.w_0_0 ctx), ((parallel_generatedRoot_215.w_1_0 ctx), ((parallel_generatedRoot_215.w_2_0 ctx), ((parallel_generatedRoot_215.w_3_0 ctx), ((parallel_generatedRoot_215.w_4_0 ctx), ((parallel_generatedRoot_215.w_5_0 ctx), ())))))) (((parallel_generatedRoot_215.w_6_0 ctx) i), (((parallel_generatedRoot_215.w_6_1 ctx) i), (((parallel_generatedRoot_215.w_6_2 ctx) i), (((parallel_generatedRoot_215.w_6_3 ctx) i), (((parallel_generatedRoot_215.w_6_4 ctx) i), (((parallel_generatedRoot_215.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_0 ctx) (0) (parallel_generatedRoot_215.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_0 ctx) (1) (parallel_generatedRoot_215.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_1 ctx) (0) (parallel_generatedRoot_215.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_1 ctx) (1) (parallel_generatedRoot_215.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_2 ctx) (0) (parallel_generatedRoot_215.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_2 ctx) (1) (parallel_generatedRoot_215.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_3 ctx) (0) (parallel_generatedRoot_215.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_3 ctx) (1) (parallel_generatedRoot_215.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_4 ctx) (0) (parallel_generatedRoot_215.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_4 ctx) (1) (parallel_generatedRoot_215.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_5 ctx) (0) (parallel_generatedRoot_215.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_215.w_6_5 ctx) (1) (parallel_generatedRoot_215.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_215_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_215.w_6_0 ctx), ((parallel_generatedRoot_215.w_6_1 ctx), ((parallel_generatedRoot_215.w_6_2 ctx), ((parallel_generatedRoot_215.w_6_3 ctx), ((parallel_generatedRoot_215.w_6_4 ctx), ((parallel_generatedRoot_215.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_215.w_9_0 ctx), ((parallel_generatedRoot_215.w_12_0 ctx), ((parallel_generatedRoot_215.w_15_0 ctx), ((parallel_generatedRoot_215.w_18_0 ctx), ((parallel_generatedRoot_215.w_21_0 ctx), ((parallel_generatedRoot_215.w_24_0 ctx), ())))))) ((parallel_generatedRoot_215.w_25_0 ctx), ((parallel_generatedRoot_215.w_25_1 ctx), ((parallel_generatedRoot_215.w_25_2 ctx), ((parallel_generatedRoot_215.w_25_3 ctx), ((parallel_generatedRoot_215.w_25_4 ctx), ((parallel_generatedRoot_215.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_215.outs ctx

theorem parallel_generatedRoot_215.facts_0  {ctx : parallel_generatedRoot_215Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_215.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_215.w_0_0 ctx) (parallel_generatedRoot_215.w_1_0 ctx) (parallel_generatedRoot_215.w_2_0 ctx) (parallel_generatedRoot_215.w_3_0 ctx) (parallel_generatedRoot_215.w_4_0 ctx) (parallel_generatedRoot_215.w_5_0 ctx) (parallel_generatedRoot_215.w_6_0 ctx) (parallel_generatedRoot_215.w_6_1 ctx) (parallel_generatedRoot_215.w_6_2 ctx) (parallel_generatedRoot_215.w_6_3 ctx) (parallel_generatedRoot_215.w_6_4 ctx) (parallel_generatedRoot_215.w_6_5 ctx) (parallel_generatedRoot_215.w_7_0 ctx) (parallel_generatedRoot_215.w_8_0 ctx) (parallel_generatedRoot_215.w_9_0 ctx) (parallel_generatedRoot_215.w_10_0 ctx) (parallel_generatedRoot_215.w_11_0 ctx) (parallel_generatedRoot_215.w_12_0 ctx) (parallel_generatedRoot_215.w_13_0 ctx) (parallel_generatedRoot_215.w_14_0 ctx) (parallel_generatedRoot_215.w_15_0 ctx) (parallel_generatedRoot_215.w_16_0 ctx) (parallel_generatedRoot_215.w_17_0 ctx) (parallel_generatedRoot_215.w_18_0 ctx) (parallel_generatedRoot_215.w_19_0 ctx) (parallel_generatedRoot_215.w_20_0 ctx) (parallel_generatedRoot_215.w_21_0 ctx) (parallel_generatedRoot_215.w_22_0 ctx) (parallel_generatedRoot_215.w_23_0 ctx) (parallel_generatedRoot_215.w_24_0 ctx) (parallel_generatedRoot_215.w_25_0 ctx) (parallel_generatedRoot_215.w_25_1 ctx) (parallel_generatedRoot_215.w_25_2 ctx) (parallel_generatedRoot_215.w_25_3 ctx) (parallel_generatedRoot_215.w_25_4 ctx) (parallel_generatedRoot_215.w_25_5 ctx) outputs) :
    parallel_generatedRoot_215.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_215.Facts  (ctx : parallel_generatedRoot_215Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_215.Facts_0 ctx outputs

theorem parallel_generatedRoot_215.facts  (ctx : parallel_generatedRoot_215Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_215.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_215.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_215.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_219.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_219.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_219Ctx" => (parallel_generatedRoot_219.Ctx backend hashModel params i_0)

def parallel_generatedRoot_219.w_0_0  (ctx : parallel_generatedRoot_219Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_219.w_1_0  (ctx : parallel_generatedRoot_219Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_219.w_2_0  (ctx : parallel_generatedRoot_219Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_219.w_3_0  (ctx : parallel_generatedRoot_219Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_219.w_4_0  (ctx : parallel_generatedRoot_219Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_219.w_5_0  (ctx : parallel_generatedRoot_219Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_219.w_6_0  (ctx : parallel_generatedRoot_219Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_219.w_6_1  (ctx : parallel_generatedRoot_219Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_219.w_6_2  (ctx : parallel_generatedRoot_219Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_219.w_6_3  (ctx : parallel_generatedRoot_219Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_219.w_6_4  (ctx : parallel_generatedRoot_219Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_219.w_6_5  (ctx : parallel_generatedRoot_219Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_219.w_7_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_219.w_8_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_219.w_9_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_219.w_7_0 ctx) (parallel_generatedRoot_219.w_8_0 ctx)

def parallel_generatedRoot_219.w_10_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_219.w_11_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_219.w_12_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_219.w_10_0 ctx) (parallel_generatedRoot_219.w_11_0 ctx)

def parallel_generatedRoot_219.w_13_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_219.w_14_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_219.w_15_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_219.w_13_0 ctx) (parallel_generatedRoot_219.w_14_0 ctx)

def parallel_generatedRoot_219.w_16_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_219.w_17_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_219.w_18_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_219.w_16_0 ctx) (parallel_generatedRoot_219.w_17_0 ctx)

def parallel_generatedRoot_219.w_19_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_219.w_20_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_219.w_21_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_219.w_19_0 ctx) (parallel_generatedRoot_219.w_20_0 ctx)

def parallel_generatedRoot_219.w_22_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_219.w_23_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_219.w_24_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_219.w_22_0 ctx) (parallel_generatedRoot_219.w_23_0 ctx)

def parallel_generatedRoot_219.w_25_0  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_219.w_25_1  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_219.w_25_2  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_219.w_25_3  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_219.w_25_4  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_219.w_25_5  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_219.outs  (ctx : parallel_generatedRoot_219Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_219.w_25_0 ctx), ((parallel_generatedRoot_219.w_25_1 ctx), ((parallel_generatedRoot_219.w_25_2 ctx), ((parallel_generatedRoot_219.w_25_3 ctx), ((parallel_generatedRoot_219.w_25_4 ctx), ((parallel_generatedRoot_219.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_219.Facts_0  (ctx : parallel_generatedRoot_219Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_219_6 backend hashModel params i_0 i ((parallel_generatedRoot_219.w_0_0 ctx), ((parallel_generatedRoot_219.w_1_0 ctx), ((parallel_generatedRoot_219.w_2_0 ctx), ((parallel_generatedRoot_219.w_3_0 ctx), ((parallel_generatedRoot_219.w_4_0 ctx), ((parallel_generatedRoot_219.w_5_0 ctx), ())))))) (((parallel_generatedRoot_219.w_6_0 ctx) i), (((parallel_generatedRoot_219.w_6_1 ctx) i), (((parallel_generatedRoot_219.w_6_2 ctx) i), (((parallel_generatedRoot_219.w_6_3 ctx) i), (((parallel_generatedRoot_219.w_6_4 ctx) i), (((parallel_generatedRoot_219.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_0 ctx) (0) (parallel_generatedRoot_219.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_0 ctx) (1) (parallel_generatedRoot_219.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_1 ctx) (0) (parallel_generatedRoot_219.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_1 ctx) (1) (parallel_generatedRoot_219.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_2 ctx) (0) (parallel_generatedRoot_219.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_2 ctx) (1) (parallel_generatedRoot_219.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_3 ctx) (0) (parallel_generatedRoot_219.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_3 ctx) (1) (parallel_generatedRoot_219.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_4 ctx) (0) (parallel_generatedRoot_219.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_4 ctx) (1) (parallel_generatedRoot_219.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_5 ctx) (0) (parallel_generatedRoot_219.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_219.w_6_5 ctx) (1) (parallel_generatedRoot_219.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_219_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_219.w_6_0 ctx), ((parallel_generatedRoot_219.w_6_1 ctx), ((parallel_generatedRoot_219.w_6_2 ctx), ((parallel_generatedRoot_219.w_6_3 ctx), ((parallel_generatedRoot_219.w_6_4 ctx), ((parallel_generatedRoot_219.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_219.w_9_0 ctx), ((parallel_generatedRoot_219.w_12_0 ctx), ((parallel_generatedRoot_219.w_15_0 ctx), ((parallel_generatedRoot_219.w_18_0 ctx), ((parallel_generatedRoot_219.w_21_0 ctx), ((parallel_generatedRoot_219.w_24_0 ctx), ())))))) ((parallel_generatedRoot_219.w_25_0 ctx), ((parallel_generatedRoot_219.w_25_1 ctx), ((parallel_generatedRoot_219.w_25_2 ctx), ((parallel_generatedRoot_219.w_25_3 ctx), ((parallel_generatedRoot_219.w_25_4 ctx), ((parallel_generatedRoot_219.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_219.outs ctx

theorem parallel_generatedRoot_219.facts_0  {ctx : parallel_generatedRoot_219Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_219.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_219.w_0_0 ctx) (parallel_generatedRoot_219.w_1_0 ctx) (parallel_generatedRoot_219.w_2_0 ctx) (parallel_generatedRoot_219.w_3_0 ctx) (parallel_generatedRoot_219.w_4_0 ctx) (parallel_generatedRoot_219.w_5_0 ctx) (parallel_generatedRoot_219.w_6_0 ctx) (parallel_generatedRoot_219.w_6_1 ctx) (parallel_generatedRoot_219.w_6_2 ctx) (parallel_generatedRoot_219.w_6_3 ctx) (parallel_generatedRoot_219.w_6_4 ctx) (parallel_generatedRoot_219.w_6_5 ctx) (parallel_generatedRoot_219.w_7_0 ctx) (parallel_generatedRoot_219.w_8_0 ctx) (parallel_generatedRoot_219.w_9_0 ctx) (parallel_generatedRoot_219.w_10_0 ctx) (parallel_generatedRoot_219.w_11_0 ctx) (parallel_generatedRoot_219.w_12_0 ctx) (parallel_generatedRoot_219.w_13_0 ctx) (parallel_generatedRoot_219.w_14_0 ctx) (parallel_generatedRoot_219.w_15_0 ctx) (parallel_generatedRoot_219.w_16_0 ctx) (parallel_generatedRoot_219.w_17_0 ctx) (parallel_generatedRoot_219.w_18_0 ctx) (parallel_generatedRoot_219.w_19_0 ctx) (parallel_generatedRoot_219.w_20_0 ctx) (parallel_generatedRoot_219.w_21_0 ctx) (parallel_generatedRoot_219.w_22_0 ctx) (parallel_generatedRoot_219.w_23_0 ctx) (parallel_generatedRoot_219.w_24_0 ctx) (parallel_generatedRoot_219.w_25_0 ctx) (parallel_generatedRoot_219.w_25_1 ctx) (parallel_generatedRoot_219.w_25_2 ctx) (parallel_generatedRoot_219.w_25_3 ctx) (parallel_generatedRoot_219.w_25_4 ctx) (parallel_generatedRoot_219.w_25_5 ctx) outputs) :
    parallel_generatedRoot_219.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_219.Facts  (ctx : parallel_generatedRoot_219Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_219.Facts_0 ctx outputs

theorem parallel_generatedRoot_219.facts  (ctx : parallel_generatedRoot_219Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_219.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_219.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_219.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_223.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_223.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_223Ctx" => (parallel_generatedRoot_223.Ctx backend hashModel params i_0)

def parallel_generatedRoot_223.w_0_0  (ctx : parallel_generatedRoot_223Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_223.w_1_0  (ctx : parallel_generatedRoot_223Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_223.w_2_0  (ctx : parallel_generatedRoot_223Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_223.w_3_0  (ctx : parallel_generatedRoot_223Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_223.w_4_0  (ctx : parallel_generatedRoot_223Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_223.w_5_0  (ctx : parallel_generatedRoot_223Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_223.w_6_0  (ctx : parallel_generatedRoot_223Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_223.w_6_1  (ctx : parallel_generatedRoot_223Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_223.w_6_2  (ctx : parallel_generatedRoot_223Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_223.w_6_3  (ctx : parallel_generatedRoot_223Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_223.w_6_4  (ctx : parallel_generatedRoot_223Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_223.w_6_5  (ctx : parallel_generatedRoot_223Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_223.w_7_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_223.w_8_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_223.w_9_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_223.w_7_0 ctx) (parallel_generatedRoot_223.w_8_0 ctx)

def parallel_generatedRoot_223.w_10_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_223.w_11_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_223.w_12_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_223.w_10_0 ctx) (parallel_generatedRoot_223.w_11_0 ctx)

def parallel_generatedRoot_223.w_13_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_223.w_14_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_223.w_15_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_223.w_13_0 ctx) (parallel_generatedRoot_223.w_14_0 ctx)

def parallel_generatedRoot_223.w_16_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_223.w_17_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_223.w_18_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_223.w_16_0 ctx) (parallel_generatedRoot_223.w_17_0 ctx)

def parallel_generatedRoot_223.w_19_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_223.w_20_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_223.w_21_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_223.w_19_0 ctx) (parallel_generatedRoot_223.w_20_0 ctx)

def parallel_generatedRoot_223.w_22_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_223.w_23_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_223.w_24_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_223.w_22_0 ctx) (parallel_generatedRoot_223.w_23_0 ctx)

def parallel_generatedRoot_223.w_25_0  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_223.w_25_1  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_223.w_25_2  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_223.w_25_3  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_223.w_25_4  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_223.w_25_5  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_223.outs  (ctx : parallel_generatedRoot_223Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_223.w_25_0 ctx), ((parallel_generatedRoot_223.w_25_1 ctx), ((parallel_generatedRoot_223.w_25_2 ctx), ((parallel_generatedRoot_223.w_25_3 ctx), ((parallel_generatedRoot_223.w_25_4 ctx), ((parallel_generatedRoot_223.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_223.Facts_0  (ctx : parallel_generatedRoot_223Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_223_6 backend hashModel params i_0 i ((parallel_generatedRoot_223.w_0_0 ctx), ((parallel_generatedRoot_223.w_1_0 ctx), ((parallel_generatedRoot_223.w_2_0 ctx), ((parallel_generatedRoot_223.w_3_0 ctx), ((parallel_generatedRoot_223.w_4_0 ctx), ((parallel_generatedRoot_223.w_5_0 ctx), ())))))) (((parallel_generatedRoot_223.w_6_0 ctx) i), (((parallel_generatedRoot_223.w_6_1 ctx) i), (((parallel_generatedRoot_223.w_6_2 ctx) i), (((parallel_generatedRoot_223.w_6_3 ctx) i), (((parallel_generatedRoot_223.w_6_4 ctx) i), (((parallel_generatedRoot_223.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_0 ctx) (0) (parallel_generatedRoot_223.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_0 ctx) (1) (parallel_generatedRoot_223.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_1 ctx) (0) (parallel_generatedRoot_223.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_1 ctx) (1) (parallel_generatedRoot_223.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_2 ctx) (0) (parallel_generatedRoot_223.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_2 ctx) (1) (parallel_generatedRoot_223.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_3 ctx) (0) (parallel_generatedRoot_223.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_3 ctx) (1) (parallel_generatedRoot_223.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_4 ctx) (0) (parallel_generatedRoot_223.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_4 ctx) (1) (parallel_generatedRoot_223.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_5 ctx) (0) (parallel_generatedRoot_223.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_223.w_6_5 ctx) (1) (parallel_generatedRoot_223.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_223_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_223.w_6_0 ctx), ((parallel_generatedRoot_223.w_6_1 ctx), ((parallel_generatedRoot_223.w_6_2 ctx), ((parallel_generatedRoot_223.w_6_3 ctx), ((parallel_generatedRoot_223.w_6_4 ctx), ((parallel_generatedRoot_223.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_223.w_9_0 ctx), ((parallel_generatedRoot_223.w_12_0 ctx), ((parallel_generatedRoot_223.w_15_0 ctx), ((parallel_generatedRoot_223.w_18_0 ctx), ((parallel_generatedRoot_223.w_21_0 ctx), ((parallel_generatedRoot_223.w_24_0 ctx), ())))))) ((parallel_generatedRoot_223.w_25_0 ctx), ((parallel_generatedRoot_223.w_25_1 ctx), ((parallel_generatedRoot_223.w_25_2 ctx), ((parallel_generatedRoot_223.w_25_3 ctx), ((parallel_generatedRoot_223.w_25_4 ctx), ((parallel_generatedRoot_223.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_223.outs ctx

theorem parallel_generatedRoot_223.facts_0  {ctx : parallel_generatedRoot_223Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_223.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_223.w_0_0 ctx) (parallel_generatedRoot_223.w_1_0 ctx) (parallel_generatedRoot_223.w_2_0 ctx) (parallel_generatedRoot_223.w_3_0 ctx) (parallel_generatedRoot_223.w_4_0 ctx) (parallel_generatedRoot_223.w_5_0 ctx) (parallel_generatedRoot_223.w_6_0 ctx) (parallel_generatedRoot_223.w_6_1 ctx) (parallel_generatedRoot_223.w_6_2 ctx) (parallel_generatedRoot_223.w_6_3 ctx) (parallel_generatedRoot_223.w_6_4 ctx) (parallel_generatedRoot_223.w_6_5 ctx) (parallel_generatedRoot_223.w_7_0 ctx) (parallel_generatedRoot_223.w_8_0 ctx) (parallel_generatedRoot_223.w_9_0 ctx) (parallel_generatedRoot_223.w_10_0 ctx) (parallel_generatedRoot_223.w_11_0 ctx) (parallel_generatedRoot_223.w_12_0 ctx) (parallel_generatedRoot_223.w_13_0 ctx) (parallel_generatedRoot_223.w_14_0 ctx) (parallel_generatedRoot_223.w_15_0 ctx) (parallel_generatedRoot_223.w_16_0 ctx) (parallel_generatedRoot_223.w_17_0 ctx) (parallel_generatedRoot_223.w_18_0 ctx) (parallel_generatedRoot_223.w_19_0 ctx) (parallel_generatedRoot_223.w_20_0 ctx) (parallel_generatedRoot_223.w_21_0 ctx) (parallel_generatedRoot_223.w_22_0 ctx) (parallel_generatedRoot_223.w_23_0 ctx) (parallel_generatedRoot_223.w_24_0 ctx) (parallel_generatedRoot_223.w_25_0 ctx) (parallel_generatedRoot_223.w_25_1 ctx) (parallel_generatedRoot_223.w_25_2 ctx) (parallel_generatedRoot_223.w_25_3 ctx) (parallel_generatedRoot_223.w_25_4 ctx) (parallel_generatedRoot_223.w_25_5 ctx) outputs) :
    parallel_generatedRoot_223.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_223.Facts  (ctx : parallel_generatedRoot_223Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_223.Facts_0 ctx outputs

theorem parallel_generatedRoot_223.facts  (ctx : parallel_generatedRoot_223Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_223.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_223.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_223.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_227.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_227.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_227Ctx" => (parallel_generatedRoot_227.Ctx backend hashModel params i_0)

def parallel_generatedRoot_227.w_0_0  (ctx : parallel_generatedRoot_227Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_227.w_1_0  (ctx : parallel_generatedRoot_227Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_227.w_2_0  (ctx : parallel_generatedRoot_227Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_227.w_3_0  (ctx : parallel_generatedRoot_227Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_227.w_4_0  (ctx : parallel_generatedRoot_227Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_227.w_5_0  (ctx : parallel_generatedRoot_227Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_227.w_6_0  (ctx : parallel_generatedRoot_227Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_227.w_6_1  (ctx : parallel_generatedRoot_227Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_227.w_6_2  (ctx : parallel_generatedRoot_227Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_227.w_6_3  (ctx : parallel_generatedRoot_227Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_227.w_6_4  (ctx : parallel_generatedRoot_227Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_227.w_6_5  (ctx : parallel_generatedRoot_227Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_227.w_7_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_227.w_8_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_227.w_9_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_227.w_7_0 ctx) (parallel_generatedRoot_227.w_8_0 ctx)

def parallel_generatedRoot_227.w_10_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_227.w_11_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_227.w_12_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_227.w_10_0 ctx) (parallel_generatedRoot_227.w_11_0 ctx)

def parallel_generatedRoot_227.w_13_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_227.w_14_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_227.w_15_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_227.w_13_0 ctx) (parallel_generatedRoot_227.w_14_0 ctx)

def parallel_generatedRoot_227.w_16_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_227.w_17_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_227.w_18_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_227.w_16_0 ctx) (parallel_generatedRoot_227.w_17_0 ctx)

def parallel_generatedRoot_227.w_19_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_227.w_20_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_227.w_21_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_227.w_19_0 ctx) (parallel_generatedRoot_227.w_20_0 ctx)

def parallel_generatedRoot_227.w_22_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_227.w_23_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_227.w_24_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_227.w_22_0 ctx) (parallel_generatedRoot_227.w_23_0 ctx)

def parallel_generatedRoot_227.w_25_0  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_227.w_25_1  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_227.w_25_2  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_227.w_25_3  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_227.w_25_4  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_227.w_25_5  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_227.outs  (ctx : parallel_generatedRoot_227Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_227.w_25_0 ctx), ((parallel_generatedRoot_227.w_25_1 ctx), ((parallel_generatedRoot_227.w_25_2 ctx), ((parallel_generatedRoot_227.w_25_3 ctx), ((parallel_generatedRoot_227.w_25_4 ctx), ((parallel_generatedRoot_227.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_227.Facts_0  (ctx : parallel_generatedRoot_227Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_227_6 backend hashModel params i_0 i ((parallel_generatedRoot_227.w_0_0 ctx), ((parallel_generatedRoot_227.w_1_0 ctx), ((parallel_generatedRoot_227.w_2_0 ctx), ((parallel_generatedRoot_227.w_3_0 ctx), ((parallel_generatedRoot_227.w_4_0 ctx), ((parallel_generatedRoot_227.w_5_0 ctx), ())))))) (((parallel_generatedRoot_227.w_6_0 ctx) i), (((parallel_generatedRoot_227.w_6_1 ctx) i), (((parallel_generatedRoot_227.w_6_2 ctx) i), (((parallel_generatedRoot_227.w_6_3 ctx) i), (((parallel_generatedRoot_227.w_6_4 ctx) i), (((parallel_generatedRoot_227.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_0 ctx) (0) (parallel_generatedRoot_227.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_0 ctx) (1) (parallel_generatedRoot_227.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_1 ctx) (0) (parallel_generatedRoot_227.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_1 ctx) (1) (parallel_generatedRoot_227.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_2 ctx) (0) (parallel_generatedRoot_227.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_2 ctx) (1) (parallel_generatedRoot_227.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_3 ctx) (0) (parallel_generatedRoot_227.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_3 ctx) (1) (parallel_generatedRoot_227.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_4 ctx) (0) (parallel_generatedRoot_227.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_4 ctx) (1) (parallel_generatedRoot_227.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_5 ctx) (0) (parallel_generatedRoot_227.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_227.w_6_5 ctx) (1) (parallel_generatedRoot_227.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_227_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_227.w_6_0 ctx), ((parallel_generatedRoot_227.w_6_1 ctx), ((parallel_generatedRoot_227.w_6_2 ctx), ((parallel_generatedRoot_227.w_6_3 ctx), ((parallel_generatedRoot_227.w_6_4 ctx), ((parallel_generatedRoot_227.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_227.w_9_0 ctx), ((parallel_generatedRoot_227.w_12_0 ctx), ((parallel_generatedRoot_227.w_15_0 ctx), ((parallel_generatedRoot_227.w_18_0 ctx), ((parallel_generatedRoot_227.w_21_0 ctx), ((parallel_generatedRoot_227.w_24_0 ctx), ())))))) ((parallel_generatedRoot_227.w_25_0 ctx), ((parallel_generatedRoot_227.w_25_1 ctx), ((parallel_generatedRoot_227.w_25_2 ctx), ((parallel_generatedRoot_227.w_25_3 ctx), ((parallel_generatedRoot_227.w_25_4 ctx), ((parallel_generatedRoot_227.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_227.outs ctx

theorem parallel_generatedRoot_227.facts_0  {ctx : parallel_generatedRoot_227Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_227.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_227.w_0_0 ctx) (parallel_generatedRoot_227.w_1_0 ctx) (parallel_generatedRoot_227.w_2_0 ctx) (parallel_generatedRoot_227.w_3_0 ctx) (parallel_generatedRoot_227.w_4_0 ctx) (parallel_generatedRoot_227.w_5_0 ctx) (parallel_generatedRoot_227.w_6_0 ctx) (parallel_generatedRoot_227.w_6_1 ctx) (parallel_generatedRoot_227.w_6_2 ctx) (parallel_generatedRoot_227.w_6_3 ctx) (parallel_generatedRoot_227.w_6_4 ctx) (parallel_generatedRoot_227.w_6_5 ctx) (parallel_generatedRoot_227.w_7_0 ctx) (parallel_generatedRoot_227.w_8_0 ctx) (parallel_generatedRoot_227.w_9_0 ctx) (parallel_generatedRoot_227.w_10_0 ctx) (parallel_generatedRoot_227.w_11_0 ctx) (parallel_generatedRoot_227.w_12_0 ctx) (parallel_generatedRoot_227.w_13_0 ctx) (parallel_generatedRoot_227.w_14_0 ctx) (parallel_generatedRoot_227.w_15_0 ctx) (parallel_generatedRoot_227.w_16_0 ctx) (parallel_generatedRoot_227.w_17_0 ctx) (parallel_generatedRoot_227.w_18_0 ctx) (parallel_generatedRoot_227.w_19_0 ctx) (parallel_generatedRoot_227.w_20_0 ctx) (parallel_generatedRoot_227.w_21_0 ctx) (parallel_generatedRoot_227.w_22_0 ctx) (parallel_generatedRoot_227.w_23_0 ctx) (parallel_generatedRoot_227.w_24_0 ctx) (parallel_generatedRoot_227.w_25_0 ctx) (parallel_generatedRoot_227.w_25_1 ctx) (parallel_generatedRoot_227.w_25_2 ctx) (parallel_generatedRoot_227.w_25_3 ctx) (parallel_generatedRoot_227.w_25_4 ctx) (parallel_generatedRoot_227.w_25_5 ctx) outputs) :
    parallel_generatedRoot_227.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_227.Facts  (ctx : parallel_generatedRoot_227Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_227.Facts_0 ctx outputs

theorem parallel_generatedRoot_227.facts  (ctx : parallel_generatedRoot_227Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_227.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_227.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_227.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_231.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_231.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_231Ctx" => (parallel_generatedRoot_231.Ctx backend hashModel params i_0)

def parallel_generatedRoot_231.w_0_0  (ctx : parallel_generatedRoot_231Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_231.w_1_0  (ctx : parallel_generatedRoot_231Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_231.w_2_0  (ctx : parallel_generatedRoot_231Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_231.w_3_0  (ctx : parallel_generatedRoot_231Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_231.w_4_0  (ctx : parallel_generatedRoot_231Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_231.w_5_0  (ctx : parallel_generatedRoot_231Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_231.w_6_0  (ctx : parallel_generatedRoot_231Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_231.w_6_1  (ctx : parallel_generatedRoot_231Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_231.w_6_2  (ctx : parallel_generatedRoot_231Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_231.w_6_3  (ctx : parallel_generatedRoot_231Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_231.w_6_4  (ctx : parallel_generatedRoot_231Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_231.w_6_5  (ctx : parallel_generatedRoot_231Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_231.w_7_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_231.w_8_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_231.w_9_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_231.w_7_0 ctx) (parallel_generatedRoot_231.w_8_0 ctx)

def parallel_generatedRoot_231.w_10_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_231.w_11_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_231.w_12_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_231.w_10_0 ctx) (parallel_generatedRoot_231.w_11_0 ctx)

def parallel_generatedRoot_231.w_13_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_231.w_14_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_231.w_15_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_231.w_13_0 ctx) (parallel_generatedRoot_231.w_14_0 ctx)

def parallel_generatedRoot_231.w_16_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_231.w_17_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_231.w_18_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_231.w_16_0 ctx) (parallel_generatedRoot_231.w_17_0 ctx)

def parallel_generatedRoot_231.w_19_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_231.w_20_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_231.w_21_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_231.w_19_0 ctx) (parallel_generatedRoot_231.w_20_0 ctx)

def parallel_generatedRoot_231.w_22_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_231.w_23_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_231.w_24_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_231.w_22_0 ctx) (parallel_generatedRoot_231.w_23_0 ctx)

def parallel_generatedRoot_231.w_25_0  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_231.w_25_1  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_231.w_25_2  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_231.w_25_3  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_231.w_25_4  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_231.w_25_5  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_231.outs  (ctx : parallel_generatedRoot_231Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_231.w_25_0 ctx), ((parallel_generatedRoot_231.w_25_1 ctx), ((parallel_generatedRoot_231.w_25_2 ctx), ((parallel_generatedRoot_231.w_25_3 ctx), ((parallel_generatedRoot_231.w_25_4 ctx), ((parallel_generatedRoot_231.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_231.Facts_0  (ctx : parallel_generatedRoot_231Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_231_6 backend hashModel params i_0 i ((parallel_generatedRoot_231.w_0_0 ctx), ((parallel_generatedRoot_231.w_1_0 ctx), ((parallel_generatedRoot_231.w_2_0 ctx), ((parallel_generatedRoot_231.w_3_0 ctx), ((parallel_generatedRoot_231.w_4_0 ctx), ((parallel_generatedRoot_231.w_5_0 ctx), ())))))) (((parallel_generatedRoot_231.w_6_0 ctx) i), (((parallel_generatedRoot_231.w_6_1 ctx) i), (((parallel_generatedRoot_231.w_6_2 ctx) i), (((parallel_generatedRoot_231.w_6_3 ctx) i), (((parallel_generatedRoot_231.w_6_4 ctx) i), (((parallel_generatedRoot_231.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_0 ctx) (0) (parallel_generatedRoot_231.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_0 ctx) (1) (parallel_generatedRoot_231.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_1 ctx) (0) (parallel_generatedRoot_231.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_1 ctx) (1) (parallel_generatedRoot_231.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_2 ctx) (0) (parallel_generatedRoot_231.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_2 ctx) (1) (parallel_generatedRoot_231.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_3 ctx) (0) (parallel_generatedRoot_231.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_3 ctx) (1) (parallel_generatedRoot_231.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_4 ctx) (0) (parallel_generatedRoot_231.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_4 ctx) (1) (parallel_generatedRoot_231.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_5 ctx) (0) (parallel_generatedRoot_231.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_231.w_6_5 ctx) (1) (parallel_generatedRoot_231.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_231_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_231.w_6_0 ctx), ((parallel_generatedRoot_231.w_6_1 ctx), ((parallel_generatedRoot_231.w_6_2 ctx), ((parallel_generatedRoot_231.w_6_3 ctx), ((parallel_generatedRoot_231.w_6_4 ctx), ((parallel_generatedRoot_231.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_231.w_9_0 ctx), ((parallel_generatedRoot_231.w_12_0 ctx), ((parallel_generatedRoot_231.w_15_0 ctx), ((parallel_generatedRoot_231.w_18_0 ctx), ((parallel_generatedRoot_231.w_21_0 ctx), ((parallel_generatedRoot_231.w_24_0 ctx), ())))))) ((parallel_generatedRoot_231.w_25_0 ctx), ((parallel_generatedRoot_231.w_25_1 ctx), ((parallel_generatedRoot_231.w_25_2 ctx), ((parallel_generatedRoot_231.w_25_3 ctx), ((parallel_generatedRoot_231.w_25_4 ctx), ((parallel_generatedRoot_231.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_231.outs ctx

theorem parallel_generatedRoot_231.facts_0  {ctx : parallel_generatedRoot_231Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_231.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_231.w_0_0 ctx) (parallel_generatedRoot_231.w_1_0 ctx) (parallel_generatedRoot_231.w_2_0 ctx) (parallel_generatedRoot_231.w_3_0 ctx) (parallel_generatedRoot_231.w_4_0 ctx) (parallel_generatedRoot_231.w_5_0 ctx) (parallel_generatedRoot_231.w_6_0 ctx) (parallel_generatedRoot_231.w_6_1 ctx) (parallel_generatedRoot_231.w_6_2 ctx) (parallel_generatedRoot_231.w_6_3 ctx) (parallel_generatedRoot_231.w_6_4 ctx) (parallel_generatedRoot_231.w_6_5 ctx) (parallel_generatedRoot_231.w_7_0 ctx) (parallel_generatedRoot_231.w_8_0 ctx) (parallel_generatedRoot_231.w_9_0 ctx) (parallel_generatedRoot_231.w_10_0 ctx) (parallel_generatedRoot_231.w_11_0 ctx) (parallel_generatedRoot_231.w_12_0 ctx) (parallel_generatedRoot_231.w_13_0 ctx) (parallel_generatedRoot_231.w_14_0 ctx) (parallel_generatedRoot_231.w_15_0 ctx) (parallel_generatedRoot_231.w_16_0 ctx) (parallel_generatedRoot_231.w_17_0 ctx) (parallel_generatedRoot_231.w_18_0 ctx) (parallel_generatedRoot_231.w_19_0 ctx) (parallel_generatedRoot_231.w_20_0 ctx) (parallel_generatedRoot_231.w_21_0 ctx) (parallel_generatedRoot_231.w_22_0 ctx) (parallel_generatedRoot_231.w_23_0 ctx) (parallel_generatedRoot_231.w_24_0 ctx) (parallel_generatedRoot_231.w_25_0 ctx) (parallel_generatedRoot_231.w_25_1 ctx) (parallel_generatedRoot_231.w_25_2 ctx) (parallel_generatedRoot_231.w_25_3 ctx) (parallel_generatedRoot_231.w_25_4 ctx) (parallel_generatedRoot_231.w_25_5 ctx) outputs) :
    parallel_generatedRoot_231.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_231.Facts  (ctx : parallel_generatedRoot_231Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_231.Facts_0 ctx outputs

theorem parallel_generatedRoot_231.facts  (ctx : parallel_generatedRoot_231Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_231.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_231.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_231.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_235.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_235.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_235Ctx" => (parallel_generatedRoot_235.Ctx backend hashModel params i_0)

def parallel_generatedRoot_235.w_0_0  (ctx : parallel_generatedRoot_235Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_235.w_1_0  (ctx : parallel_generatedRoot_235Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_235.w_2_0  (ctx : parallel_generatedRoot_235Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_235.w_3_0  (ctx : parallel_generatedRoot_235Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_235.w_4_0  (ctx : parallel_generatedRoot_235Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_235.w_5_0  (ctx : parallel_generatedRoot_235Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_235.w_6_0  (ctx : parallel_generatedRoot_235Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_235.w_6_1  (ctx : parallel_generatedRoot_235Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_235.w_6_2  (ctx : parallel_generatedRoot_235Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_235.w_6_3  (ctx : parallel_generatedRoot_235Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_235.w_6_4  (ctx : parallel_generatedRoot_235Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_235.w_6_5  (ctx : parallel_generatedRoot_235Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_235.w_7_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_235.w_8_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_235.w_9_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_235.w_7_0 ctx) (parallel_generatedRoot_235.w_8_0 ctx)

def parallel_generatedRoot_235.w_10_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_235.w_11_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_235.w_12_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_235.w_10_0 ctx) (parallel_generatedRoot_235.w_11_0 ctx)

def parallel_generatedRoot_235.w_13_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_235.w_14_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_235.w_15_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_235.w_13_0 ctx) (parallel_generatedRoot_235.w_14_0 ctx)

def parallel_generatedRoot_235.w_16_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_235.w_17_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_235.w_18_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_235.w_16_0 ctx) (parallel_generatedRoot_235.w_17_0 ctx)

def parallel_generatedRoot_235.w_19_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_235.w_20_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_235.w_21_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_235.w_19_0 ctx) (parallel_generatedRoot_235.w_20_0 ctx)

def parallel_generatedRoot_235.w_22_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_235.w_23_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_235.w_24_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_235.w_22_0 ctx) (parallel_generatedRoot_235.w_23_0 ctx)

def parallel_generatedRoot_235.w_25_0  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_235.w_25_1  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_235.w_25_2  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_235.w_25_3  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_235.w_25_4  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_235.w_25_5  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_235.outs  (ctx : parallel_generatedRoot_235Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_235.w_25_0 ctx), ((parallel_generatedRoot_235.w_25_1 ctx), ((parallel_generatedRoot_235.w_25_2 ctx), ((parallel_generatedRoot_235.w_25_3 ctx), ((parallel_generatedRoot_235.w_25_4 ctx), ((parallel_generatedRoot_235.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_235.Facts_0  (ctx : parallel_generatedRoot_235Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_235_6 backend hashModel params i_0 i ((parallel_generatedRoot_235.w_0_0 ctx), ((parallel_generatedRoot_235.w_1_0 ctx), ((parallel_generatedRoot_235.w_2_0 ctx), ((parallel_generatedRoot_235.w_3_0 ctx), ((parallel_generatedRoot_235.w_4_0 ctx), ((parallel_generatedRoot_235.w_5_0 ctx), ())))))) (((parallel_generatedRoot_235.w_6_0 ctx) i), (((parallel_generatedRoot_235.w_6_1 ctx) i), (((parallel_generatedRoot_235.w_6_2 ctx) i), (((parallel_generatedRoot_235.w_6_3 ctx) i), (((parallel_generatedRoot_235.w_6_4 ctx) i), (((parallel_generatedRoot_235.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_0 ctx) (0) (parallel_generatedRoot_235.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_0 ctx) (1) (parallel_generatedRoot_235.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_1 ctx) (0) (parallel_generatedRoot_235.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_1 ctx) (1) (parallel_generatedRoot_235.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_2 ctx) (0) (parallel_generatedRoot_235.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_2 ctx) (1) (parallel_generatedRoot_235.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_3 ctx) (0) (parallel_generatedRoot_235.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_3 ctx) (1) (parallel_generatedRoot_235.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_4 ctx) (0) (parallel_generatedRoot_235.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_4 ctx) (1) (parallel_generatedRoot_235.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_5 ctx) (0) (parallel_generatedRoot_235.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_235.w_6_5 ctx) (1) (parallel_generatedRoot_235.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_235_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_235.w_6_0 ctx), ((parallel_generatedRoot_235.w_6_1 ctx), ((parallel_generatedRoot_235.w_6_2 ctx), ((parallel_generatedRoot_235.w_6_3 ctx), ((parallel_generatedRoot_235.w_6_4 ctx), ((parallel_generatedRoot_235.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_235.w_9_0 ctx), ((parallel_generatedRoot_235.w_12_0 ctx), ((parallel_generatedRoot_235.w_15_0 ctx), ((parallel_generatedRoot_235.w_18_0 ctx), ((parallel_generatedRoot_235.w_21_0 ctx), ((parallel_generatedRoot_235.w_24_0 ctx), ())))))) ((parallel_generatedRoot_235.w_25_0 ctx), ((parallel_generatedRoot_235.w_25_1 ctx), ((parallel_generatedRoot_235.w_25_2 ctx), ((parallel_generatedRoot_235.w_25_3 ctx), ((parallel_generatedRoot_235.w_25_4 ctx), ((parallel_generatedRoot_235.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_235.outs ctx

theorem parallel_generatedRoot_235.facts_0  {ctx : parallel_generatedRoot_235Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_235.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_235.w_0_0 ctx) (parallel_generatedRoot_235.w_1_0 ctx) (parallel_generatedRoot_235.w_2_0 ctx) (parallel_generatedRoot_235.w_3_0 ctx) (parallel_generatedRoot_235.w_4_0 ctx) (parallel_generatedRoot_235.w_5_0 ctx) (parallel_generatedRoot_235.w_6_0 ctx) (parallel_generatedRoot_235.w_6_1 ctx) (parallel_generatedRoot_235.w_6_2 ctx) (parallel_generatedRoot_235.w_6_3 ctx) (parallel_generatedRoot_235.w_6_4 ctx) (parallel_generatedRoot_235.w_6_5 ctx) (parallel_generatedRoot_235.w_7_0 ctx) (parallel_generatedRoot_235.w_8_0 ctx) (parallel_generatedRoot_235.w_9_0 ctx) (parallel_generatedRoot_235.w_10_0 ctx) (parallel_generatedRoot_235.w_11_0 ctx) (parallel_generatedRoot_235.w_12_0 ctx) (parallel_generatedRoot_235.w_13_0 ctx) (parallel_generatedRoot_235.w_14_0 ctx) (parallel_generatedRoot_235.w_15_0 ctx) (parallel_generatedRoot_235.w_16_0 ctx) (parallel_generatedRoot_235.w_17_0 ctx) (parallel_generatedRoot_235.w_18_0 ctx) (parallel_generatedRoot_235.w_19_0 ctx) (parallel_generatedRoot_235.w_20_0 ctx) (parallel_generatedRoot_235.w_21_0 ctx) (parallel_generatedRoot_235.w_22_0 ctx) (parallel_generatedRoot_235.w_23_0 ctx) (parallel_generatedRoot_235.w_24_0 ctx) (parallel_generatedRoot_235.w_25_0 ctx) (parallel_generatedRoot_235.w_25_1 ctx) (parallel_generatedRoot_235.w_25_2 ctx) (parallel_generatedRoot_235.w_25_3 ctx) (parallel_generatedRoot_235.w_25_4 ctx) (parallel_generatedRoot_235.w_25_5 ctx) outputs) :
    parallel_generatedRoot_235.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_235.Facts  (ctx : parallel_generatedRoot_235Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_235.Facts_0 ctx outputs

theorem parallel_generatedRoot_235.facts  (ctx : parallel_generatedRoot_235Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_235.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_235.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_235.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_239.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_239.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_239Ctx" => (parallel_generatedRoot_239.Ctx backend hashModel params i_0)

def parallel_generatedRoot_239.w_0_0  (ctx : parallel_generatedRoot_239Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_239.w_1_0  (ctx : parallel_generatedRoot_239Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_239.w_2_0  (ctx : parallel_generatedRoot_239Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_239.w_3_0  (ctx : parallel_generatedRoot_239Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_239.w_4_0  (ctx : parallel_generatedRoot_239Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_239.w_5_0  (ctx : parallel_generatedRoot_239Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_239.w_6_0  (ctx : parallel_generatedRoot_239Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_239.w_6_1  (ctx : parallel_generatedRoot_239Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_239.w_6_2  (ctx : parallel_generatedRoot_239Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_239.w_6_3  (ctx : parallel_generatedRoot_239Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_239.w_6_4  (ctx : parallel_generatedRoot_239Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_239.w_6_5  (ctx : parallel_generatedRoot_239Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_239.w_7_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_239.w_8_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_239.w_9_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_239.w_7_0 ctx) (parallel_generatedRoot_239.w_8_0 ctx)

def parallel_generatedRoot_239.w_10_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_239.w_11_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_239.w_12_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_239.w_10_0 ctx) (parallel_generatedRoot_239.w_11_0 ctx)

def parallel_generatedRoot_239.w_13_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_239.w_14_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_239.w_15_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_239.w_13_0 ctx) (parallel_generatedRoot_239.w_14_0 ctx)

def parallel_generatedRoot_239.w_16_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_239.w_17_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_239.w_18_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_239.w_16_0 ctx) (parallel_generatedRoot_239.w_17_0 ctx)

def parallel_generatedRoot_239.w_19_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_239.w_20_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_239.w_21_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_239.w_19_0 ctx) (parallel_generatedRoot_239.w_20_0 ctx)

def parallel_generatedRoot_239.w_22_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_239.w_23_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_239.w_24_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_239.w_22_0 ctx) (parallel_generatedRoot_239.w_23_0 ctx)

def parallel_generatedRoot_239.w_25_0  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_239.w_25_1  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_239.w_25_2  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_239.w_25_3  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_239.w_25_4  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_239.w_25_5  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_239.outs  (ctx : parallel_generatedRoot_239Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_239.w_25_0 ctx), ((parallel_generatedRoot_239.w_25_1 ctx), ((parallel_generatedRoot_239.w_25_2 ctx), ((parallel_generatedRoot_239.w_25_3 ctx), ((parallel_generatedRoot_239.w_25_4 ctx), ((parallel_generatedRoot_239.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_239.Facts_0  (ctx : parallel_generatedRoot_239Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_239_6 backend hashModel params i_0 i ((parallel_generatedRoot_239.w_0_0 ctx), ((parallel_generatedRoot_239.w_1_0 ctx), ((parallel_generatedRoot_239.w_2_0 ctx), ((parallel_generatedRoot_239.w_3_0 ctx), ((parallel_generatedRoot_239.w_4_0 ctx), ((parallel_generatedRoot_239.w_5_0 ctx), ())))))) (((parallel_generatedRoot_239.w_6_0 ctx) i), (((parallel_generatedRoot_239.w_6_1 ctx) i), (((parallel_generatedRoot_239.w_6_2 ctx) i), (((parallel_generatedRoot_239.w_6_3 ctx) i), (((parallel_generatedRoot_239.w_6_4 ctx) i), (((parallel_generatedRoot_239.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_0 ctx) (0) (parallel_generatedRoot_239.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_0 ctx) (1) (parallel_generatedRoot_239.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_1 ctx) (0) (parallel_generatedRoot_239.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_1 ctx) (1) (parallel_generatedRoot_239.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_2 ctx) (0) (parallel_generatedRoot_239.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_2 ctx) (1) (parallel_generatedRoot_239.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_3 ctx) (0) (parallel_generatedRoot_239.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_3 ctx) (1) (parallel_generatedRoot_239.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_4 ctx) (0) (parallel_generatedRoot_239.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_4 ctx) (1) (parallel_generatedRoot_239.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_5 ctx) (0) (parallel_generatedRoot_239.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_239.w_6_5 ctx) (1) (parallel_generatedRoot_239.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_239_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_239.w_6_0 ctx), ((parallel_generatedRoot_239.w_6_1 ctx), ((parallel_generatedRoot_239.w_6_2 ctx), ((parallel_generatedRoot_239.w_6_3 ctx), ((parallel_generatedRoot_239.w_6_4 ctx), ((parallel_generatedRoot_239.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_239.w_9_0 ctx), ((parallel_generatedRoot_239.w_12_0 ctx), ((parallel_generatedRoot_239.w_15_0 ctx), ((parallel_generatedRoot_239.w_18_0 ctx), ((parallel_generatedRoot_239.w_21_0 ctx), ((parallel_generatedRoot_239.w_24_0 ctx), ())))))) ((parallel_generatedRoot_239.w_25_0 ctx), ((parallel_generatedRoot_239.w_25_1 ctx), ((parallel_generatedRoot_239.w_25_2 ctx), ((parallel_generatedRoot_239.w_25_3 ctx), ((parallel_generatedRoot_239.w_25_4 ctx), ((parallel_generatedRoot_239.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_239.outs ctx

theorem parallel_generatedRoot_239.facts_0  {ctx : parallel_generatedRoot_239Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_239.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_239.w_0_0 ctx) (parallel_generatedRoot_239.w_1_0 ctx) (parallel_generatedRoot_239.w_2_0 ctx) (parallel_generatedRoot_239.w_3_0 ctx) (parallel_generatedRoot_239.w_4_0 ctx) (parallel_generatedRoot_239.w_5_0 ctx) (parallel_generatedRoot_239.w_6_0 ctx) (parallel_generatedRoot_239.w_6_1 ctx) (parallel_generatedRoot_239.w_6_2 ctx) (parallel_generatedRoot_239.w_6_3 ctx) (parallel_generatedRoot_239.w_6_4 ctx) (parallel_generatedRoot_239.w_6_5 ctx) (parallel_generatedRoot_239.w_7_0 ctx) (parallel_generatedRoot_239.w_8_0 ctx) (parallel_generatedRoot_239.w_9_0 ctx) (parallel_generatedRoot_239.w_10_0 ctx) (parallel_generatedRoot_239.w_11_0 ctx) (parallel_generatedRoot_239.w_12_0 ctx) (parallel_generatedRoot_239.w_13_0 ctx) (parallel_generatedRoot_239.w_14_0 ctx) (parallel_generatedRoot_239.w_15_0 ctx) (parallel_generatedRoot_239.w_16_0 ctx) (parallel_generatedRoot_239.w_17_0 ctx) (parallel_generatedRoot_239.w_18_0 ctx) (parallel_generatedRoot_239.w_19_0 ctx) (parallel_generatedRoot_239.w_20_0 ctx) (parallel_generatedRoot_239.w_21_0 ctx) (parallel_generatedRoot_239.w_22_0 ctx) (parallel_generatedRoot_239.w_23_0 ctx) (parallel_generatedRoot_239.w_24_0 ctx) (parallel_generatedRoot_239.w_25_0 ctx) (parallel_generatedRoot_239.w_25_1 ctx) (parallel_generatedRoot_239.w_25_2 ctx) (parallel_generatedRoot_239.w_25_3 ctx) (parallel_generatedRoot_239.w_25_4 ctx) (parallel_generatedRoot_239.w_25_5 ctx) outputs) :
    parallel_generatedRoot_239.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_239.Facts  (ctx : parallel_generatedRoot_239Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_239.Facts_0 ctx outputs

theorem parallel_generatedRoot_239.facts  (ctx : parallel_generatedRoot_239Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_239.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_239.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_239.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_243.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_243.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_243Ctx" => (parallel_generatedRoot_243.Ctx backend hashModel params i_0)

def parallel_generatedRoot_243.w_0_0  (ctx : parallel_generatedRoot_243Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_243.w_1_0  (ctx : parallel_generatedRoot_243Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_243.w_2_0  (ctx : parallel_generatedRoot_243Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_243.w_3_0  (ctx : parallel_generatedRoot_243Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_243.w_4_0  (ctx : parallel_generatedRoot_243Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_243.w_5_0  (ctx : parallel_generatedRoot_243Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_243.w_6_0  (ctx : parallel_generatedRoot_243Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_243.w_6_1  (ctx : parallel_generatedRoot_243Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_243.w_6_2  (ctx : parallel_generatedRoot_243Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_243.w_6_3  (ctx : parallel_generatedRoot_243Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_243.w_6_4  (ctx : parallel_generatedRoot_243Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_243.w_6_5  (ctx : parallel_generatedRoot_243Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_243.w_7_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_243.w_8_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_243.w_9_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_243.w_7_0 ctx) (parallel_generatedRoot_243.w_8_0 ctx)

def parallel_generatedRoot_243.w_10_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_243.w_11_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_243.w_12_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_243.w_10_0 ctx) (parallel_generatedRoot_243.w_11_0 ctx)

def parallel_generatedRoot_243.w_13_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_243.w_14_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_243.w_15_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_243.w_13_0 ctx) (parallel_generatedRoot_243.w_14_0 ctx)

def parallel_generatedRoot_243.w_16_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_243.w_17_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_243.w_18_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_243.w_16_0 ctx) (parallel_generatedRoot_243.w_17_0 ctx)

def parallel_generatedRoot_243.w_19_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_243.w_20_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_243.w_21_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_243.w_19_0 ctx) (parallel_generatedRoot_243.w_20_0 ctx)

def parallel_generatedRoot_243.w_22_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_243.w_23_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_243.w_24_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_243.w_22_0 ctx) (parallel_generatedRoot_243.w_23_0 ctx)

def parallel_generatedRoot_243.w_25_0  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_243.w_25_1  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_243.w_25_2  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_243.w_25_3  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_243.w_25_4  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_243.w_25_5  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_243.outs  (ctx : parallel_generatedRoot_243Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_243.w_25_0 ctx), ((parallel_generatedRoot_243.w_25_1 ctx), ((parallel_generatedRoot_243.w_25_2 ctx), ((parallel_generatedRoot_243.w_25_3 ctx), ((parallel_generatedRoot_243.w_25_4 ctx), ((parallel_generatedRoot_243.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_243.Facts_0  (ctx : parallel_generatedRoot_243Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_243_6 backend hashModel params i_0 i ((parallel_generatedRoot_243.w_0_0 ctx), ((parallel_generatedRoot_243.w_1_0 ctx), ((parallel_generatedRoot_243.w_2_0 ctx), ((parallel_generatedRoot_243.w_3_0 ctx), ((parallel_generatedRoot_243.w_4_0 ctx), ((parallel_generatedRoot_243.w_5_0 ctx), ())))))) (((parallel_generatedRoot_243.w_6_0 ctx) i), (((parallel_generatedRoot_243.w_6_1 ctx) i), (((parallel_generatedRoot_243.w_6_2 ctx) i), (((parallel_generatedRoot_243.w_6_3 ctx) i), (((parallel_generatedRoot_243.w_6_4 ctx) i), (((parallel_generatedRoot_243.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_0 ctx) (0) (parallel_generatedRoot_243.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_0 ctx) (1) (parallel_generatedRoot_243.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_1 ctx) (0) (parallel_generatedRoot_243.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_1 ctx) (1) (parallel_generatedRoot_243.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_2 ctx) (0) (parallel_generatedRoot_243.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_2 ctx) (1) (parallel_generatedRoot_243.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_3 ctx) (0) (parallel_generatedRoot_243.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_3 ctx) (1) (parallel_generatedRoot_243.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_4 ctx) (0) (parallel_generatedRoot_243.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_4 ctx) (1) (parallel_generatedRoot_243.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_5 ctx) (0) (parallel_generatedRoot_243.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_243.w_6_5 ctx) (1) (parallel_generatedRoot_243.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_243_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_243.w_6_0 ctx), ((parallel_generatedRoot_243.w_6_1 ctx), ((parallel_generatedRoot_243.w_6_2 ctx), ((parallel_generatedRoot_243.w_6_3 ctx), ((parallel_generatedRoot_243.w_6_4 ctx), ((parallel_generatedRoot_243.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_243.w_9_0 ctx), ((parallel_generatedRoot_243.w_12_0 ctx), ((parallel_generatedRoot_243.w_15_0 ctx), ((parallel_generatedRoot_243.w_18_0 ctx), ((parallel_generatedRoot_243.w_21_0 ctx), ((parallel_generatedRoot_243.w_24_0 ctx), ())))))) ((parallel_generatedRoot_243.w_25_0 ctx), ((parallel_generatedRoot_243.w_25_1 ctx), ((parallel_generatedRoot_243.w_25_2 ctx), ((parallel_generatedRoot_243.w_25_3 ctx), ((parallel_generatedRoot_243.w_25_4 ctx), ((parallel_generatedRoot_243.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_243.outs ctx

theorem parallel_generatedRoot_243.facts_0  {ctx : parallel_generatedRoot_243Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_243.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_243.w_0_0 ctx) (parallel_generatedRoot_243.w_1_0 ctx) (parallel_generatedRoot_243.w_2_0 ctx) (parallel_generatedRoot_243.w_3_0 ctx) (parallel_generatedRoot_243.w_4_0 ctx) (parallel_generatedRoot_243.w_5_0 ctx) (parallel_generatedRoot_243.w_6_0 ctx) (parallel_generatedRoot_243.w_6_1 ctx) (parallel_generatedRoot_243.w_6_2 ctx) (parallel_generatedRoot_243.w_6_3 ctx) (parallel_generatedRoot_243.w_6_4 ctx) (parallel_generatedRoot_243.w_6_5 ctx) (parallel_generatedRoot_243.w_7_0 ctx) (parallel_generatedRoot_243.w_8_0 ctx) (parallel_generatedRoot_243.w_9_0 ctx) (parallel_generatedRoot_243.w_10_0 ctx) (parallel_generatedRoot_243.w_11_0 ctx) (parallel_generatedRoot_243.w_12_0 ctx) (parallel_generatedRoot_243.w_13_0 ctx) (parallel_generatedRoot_243.w_14_0 ctx) (parallel_generatedRoot_243.w_15_0 ctx) (parallel_generatedRoot_243.w_16_0 ctx) (parallel_generatedRoot_243.w_17_0 ctx) (parallel_generatedRoot_243.w_18_0 ctx) (parallel_generatedRoot_243.w_19_0 ctx) (parallel_generatedRoot_243.w_20_0 ctx) (parallel_generatedRoot_243.w_21_0 ctx) (parallel_generatedRoot_243.w_22_0 ctx) (parallel_generatedRoot_243.w_23_0 ctx) (parallel_generatedRoot_243.w_24_0 ctx) (parallel_generatedRoot_243.w_25_0 ctx) (parallel_generatedRoot_243.w_25_1 ctx) (parallel_generatedRoot_243.w_25_2 ctx) (parallel_generatedRoot_243.w_25_3 ctx) (parallel_generatedRoot_243.w_25_4 ctx) (parallel_generatedRoot_243.w_25_5 ctx) outputs) :
    parallel_generatedRoot_243.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_243.Facts  (ctx : parallel_generatedRoot_243Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_243.Facts_0 ctx outputs

theorem parallel_generatedRoot_243.facts  (ctx : parallel_generatedRoot_243Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_243.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_243.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_243.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_247.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_247.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_247Ctx" => (parallel_generatedRoot_247.Ctx backend hashModel params i_0)

def parallel_generatedRoot_247.w_0_0  (ctx : parallel_generatedRoot_247Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_247.w_1_0  (ctx : parallel_generatedRoot_247Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_247.w_2_0  (ctx : parallel_generatedRoot_247Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_247.w_3_0  (ctx : parallel_generatedRoot_247Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_247.w_4_0  (ctx : parallel_generatedRoot_247Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_247.w_5_0  (ctx : parallel_generatedRoot_247Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_247.w_6_0  (ctx : parallel_generatedRoot_247Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_247.w_6_1  (ctx : parallel_generatedRoot_247Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_247.w_6_2  (ctx : parallel_generatedRoot_247Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_247.w_6_3  (ctx : parallel_generatedRoot_247Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_247.w_6_4  (ctx : parallel_generatedRoot_247Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_247.w_6_5  (ctx : parallel_generatedRoot_247Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_247.w_7_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_247.w_8_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_247.w_9_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_247.w_7_0 ctx) (parallel_generatedRoot_247.w_8_0 ctx)

def parallel_generatedRoot_247.w_10_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_247.w_11_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_247.w_12_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_247.w_10_0 ctx) (parallel_generatedRoot_247.w_11_0 ctx)

def parallel_generatedRoot_247.w_13_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_247.w_14_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_247.w_15_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_247.w_13_0 ctx) (parallel_generatedRoot_247.w_14_0 ctx)

def parallel_generatedRoot_247.w_16_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_247.w_17_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_247.w_18_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_247.w_16_0 ctx) (parallel_generatedRoot_247.w_17_0 ctx)

def parallel_generatedRoot_247.w_19_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_247.w_20_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_247.w_21_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_247.w_19_0 ctx) (parallel_generatedRoot_247.w_20_0 ctx)

def parallel_generatedRoot_247.w_22_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_247.w_23_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_247.w_24_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_247.w_22_0 ctx) (parallel_generatedRoot_247.w_23_0 ctx)

def parallel_generatedRoot_247.w_25_0  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_247.w_25_1  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_247.w_25_2  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_247.w_25_3  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_247.w_25_4  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_247.w_25_5  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_247.outs  (ctx : parallel_generatedRoot_247Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_247.w_25_0 ctx), ((parallel_generatedRoot_247.w_25_1 ctx), ((parallel_generatedRoot_247.w_25_2 ctx), ((parallel_generatedRoot_247.w_25_3 ctx), ((parallel_generatedRoot_247.w_25_4 ctx), ((parallel_generatedRoot_247.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_247.Facts_0  (ctx : parallel_generatedRoot_247Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_247_6 backend hashModel params i_0 i ((parallel_generatedRoot_247.w_0_0 ctx), ((parallel_generatedRoot_247.w_1_0 ctx), ((parallel_generatedRoot_247.w_2_0 ctx), ((parallel_generatedRoot_247.w_3_0 ctx), ((parallel_generatedRoot_247.w_4_0 ctx), ((parallel_generatedRoot_247.w_5_0 ctx), ())))))) (((parallel_generatedRoot_247.w_6_0 ctx) i), (((parallel_generatedRoot_247.w_6_1 ctx) i), (((parallel_generatedRoot_247.w_6_2 ctx) i), (((parallel_generatedRoot_247.w_6_3 ctx) i), (((parallel_generatedRoot_247.w_6_4 ctx) i), (((parallel_generatedRoot_247.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_0 ctx) (0) (parallel_generatedRoot_247.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_0 ctx) (1) (parallel_generatedRoot_247.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_1 ctx) (0) (parallel_generatedRoot_247.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_1 ctx) (1) (parallel_generatedRoot_247.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_2 ctx) (0) (parallel_generatedRoot_247.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_2 ctx) (1) (parallel_generatedRoot_247.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_3 ctx) (0) (parallel_generatedRoot_247.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_3 ctx) (1) (parallel_generatedRoot_247.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_4 ctx) (0) (parallel_generatedRoot_247.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_4 ctx) (1) (parallel_generatedRoot_247.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_5 ctx) (0) (parallel_generatedRoot_247.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_247.w_6_5 ctx) (1) (parallel_generatedRoot_247.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_247_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_247.w_6_0 ctx), ((parallel_generatedRoot_247.w_6_1 ctx), ((parallel_generatedRoot_247.w_6_2 ctx), ((parallel_generatedRoot_247.w_6_3 ctx), ((parallel_generatedRoot_247.w_6_4 ctx), ((parallel_generatedRoot_247.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_247.w_9_0 ctx), ((parallel_generatedRoot_247.w_12_0 ctx), ((parallel_generatedRoot_247.w_15_0 ctx), ((parallel_generatedRoot_247.w_18_0 ctx), ((parallel_generatedRoot_247.w_21_0 ctx), ((parallel_generatedRoot_247.w_24_0 ctx), ())))))) ((parallel_generatedRoot_247.w_25_0 ctx), ((parallel_generatedRoot_247.w_25_1 ctx), ((parallel_generatedRoot_247.w_25_2 ctx), ((parallel_generatedRoot_247.w_25_3 ctx), ((parallel_generatedRoot_247.w_25_4 ctx), ((parallel_generatedRoot_247.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_247.outs ctx

theorem parallel_generatedRoot_247.facts_0  {ctx : parallel_generatedRoot_247Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_247.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_247.w_0_0 ctx) (parallel_generatedRoot_247.w_1_0 ctx) (parallel_generatedRoot_247.w_2_0 ctx) (parallel_generatedRoot_247.w_3_0 ctx) (parallel_generatedRoot_247.w_4_0 ctx) (parallel_generatedRoot_247.w_5_0 ctx) (parallel_generatedRoot_247.w_6_0 ctx) (parallel_generatedRoot_247.w_6_1 ctx) (parallel_generatedRoot_247.w_6_2 ctx) (parallel_generatedRoot_247.w_6_3 ctx) (parallel_generatedRoot_247.w_6_4 ctx) (parallel_generatedRoot_247.w_6_5 ctx) (parallel_generatedRoot_247.w_7_0 ctx) (parallel_generatedRoot_247.w_8_0 ctx) (parallel_generatedRoot_247.w_9_0 ctx) (parallel_generatedRoot_247.w_10_0 ctx) (parallel_generatedRoot_247.w_11_0 ctx) (parallel_generatedRoot_247.w_12_0 ctx) (parallel_generatedRoot_247.w_13_0 ctx) (parallel_generatedRoot_247.w_14_0 ctx) (parallel_generatedRoot_247.w_15_0 ctx) (parallel_generatedRoot_247.w_16_0 ctx) (parallel_generatedRoot_247.w_17_0 ctx) (parallel_generatedRoot_247.w_18_0 ctx) (parallel_generatedRoot_247.w_19_0 ctx) (parallel_generatedRoot_247.w_20_0 ctx) (parallel_generatedRoot_247.w_21_0 ctx) (parallel_generatedRoot_247.w_22_0 ctx) (parallel_generatedRoot_247.w_23_0 ctx) (parallel_generatedRoot_247.w_24_0 ctx) (parallel_generatedRoot_247.w_25_0 ctx) (parallel_generatedRoot_247.w_25_1 ctx) (parallel_generatedRoot_247.w_25_2 ctx) (parallel_generatedRoot_247.w_25_3 ctx) (parallel_generatedRoot_247.w_25_4 ctx) (parallel_generatedRoot_247.w_25_5 ctx) outputs) :
    parallel_generatedRoot_247.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_247.Facts  (ctx : parallel_generatedRoot_247Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_247.Facts_0 ctx outputs

theorem parallel_generatedRoot_247.facts  (ctx : parallel_generatedRoot_247Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_247.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_247.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_247.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_251.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_251.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_251Ctx" => (parallel_generatedRoot_251.Ctx backend hashModel params i_0)

def parallel_generatedRoot_251.w_0_0  (ctx : parallel_generatedRoot_251Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_251.w_1_0  (ctx : parallel_generatedRoot_251Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_251.w_2_0  (ctx : parallel_generatedRoot_251Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_251.w_3_0  (ctx : parallel_generatedRoot_251Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_251.w_4_0  (ctx : parallel_generatedRoot_251Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_251.w_5_0  (ctx : parallel_generatedRoot_251Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_251.w_6_0  (ctx : parallel_generatedRoot_251Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_251.w_6_1  (ctx : parallel_generatedRoot_251Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_251.w_6_2  (ctx : parallel_generatedRoot_251Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_251.w_6_3  (ctx : parallel_generatedRoot_251Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_251.w_6_4  (ctx : parallel_generatedRoot_251Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_251.w_6_5  (ctx : parallel_generatedRoot_251Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_251.w_7_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_251.w_8_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_251.w_9_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_251.w_7_0 ctx) (parallel_generatedRoot_251.w_8_0 ctx)

def parallel_generatedRoot_251.w_10_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_251.w_11_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_251.w_12_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_251.w_10_0 ctx) (parallel_generatedRoot_251.w_11_0 ctx)

def parallel_generatedRoot_251.w_13_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_251.w_14_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_251.w_15_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_251.w_13_0 ctx) (parallel_generatedRoot_251.w_14_0 ctx)

def parallel_generatedRoot_251.w_16_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_251.w_17_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_251.w_18_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_251.w_16_0 ctx) (parallel_generatedRoot_251.w_17_0 ctx)

def parallel_generatedRoot_251.w_19_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_251.w_20_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_251.w_21_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_251.w_19_0 ctx) (parallel_generatedRoot_251.w_20_0 ctx)

def parallel_generatedRoot_251.w_22_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_251.w_23_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_251.w_24_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_251.w_22_0 ctx) (parallel_generatedRoot_251.w_23_0 ctx)

def parallel_generatedRoot_251.w_25_0  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_251.w_25_1  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_251.w_25_2  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_251.w_25_3  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_251.w_25_4  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_251.w_25_5  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_251.outs  (ctx : parallel_generatedRoot_251Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_251.w_25_0 ctx), ((parallel_generatedRoot_251.w_25_1 ctx), ((parallel_generatedRoot_251.w_25_2 ctx), ((parallel_generatedRoot_251.w_25_3 ctx), ((parallel_generatedRoot_251.w_25_4 ctx), ((parallel_generatedRoot_251.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_251.Facts_0  (ctx : parallel_generatedRoot_251Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_251_6 backend hashModel params i_0 i ((parallel_generatedRoot_251.w_0_0 ctx), ((parallel_generatedRoot_251.w_1_0 ctx), ((parallel_generatedRoot_251.w_2_0 ctx), ((parallel_generatedRoot_251.w_3_0 ctx), ((parallel_generatedRoot_251.w_4_0 ctx), ((parallel_generatedRoot_251.w_5_0 ctx), ())))))) (((parallel_generatedRoot_251.w_6_0 ctx) i), (((parallel_generatedRoot_251.w_6_1 ctx) i), (((parallel_generatedRoot_251.w_6_2 ctx) i), (((parallel_generatedRoot_251.w_6_3 ctx) i), (((parallel_generatedRoot_251.w_6_4 ctx) i), (((parallel_generatedRoot_251.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_0 ctx) (0) (parallel_generatedRoot_251.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_0 ctx) (1) (parallel_generatedRoot_251.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_1 ctx) (0) (parallel_generatedRoot_251.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_1 ctx) (1) (parallel_generatedRoot_251.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_2 ctx) (0) (parallel_generatedRoot_251.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_2 ctx) (1) (parallel_generatedRoot_251.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_3 ctx) (0) (parallel_generatedRoot_251.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_3 ctx) (1) (parallel_generatedRoot_251.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_4 ctx) (0) (parallel_generatedRoot_251.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_4 ctx) (1) (parallel_generatedRoot_251.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_5 ctx) (0) (parallel_generatedRoot_251.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_251.w_6_5 ctx) (1) (parallel_generatedRoot_251.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_251_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_251.w_6_0 ctx), ((parallel_generatedRoot_251.w_6_1 ctx), ((parallel_generatedRoot_251.w_6_2 ctx), ((parallel_generatedRoot_251.w_6_3 ctx), ((parallel_generatedRoot_251.w_6_4 ctx), ((parallel_generatedRoot_251.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_251.w_9_0 ctx), ((parallel_generatedRoot_251.w_12_0 ctx), ((parallel_generatedRoot_251.w_15_0 ctx), ((parallel_generatedRoot_251.w_18_0 ctx), ((parallel_generatedRoot_251.w_21_0 ctx), ((parallel_generatedRoot_251.w_24_0 ctx), ())))))) ((parallel_generatedRoot_251.w_25_0 ctx), ((parallel_generatedRoot_251.w_25_1 ctx), ((parallel_generatedRoot_251.w_25_2 ctx), ((parallel_generatedRoot_251.w_25_3 ctx), ((parallel_generatedRoot_251.w_25_4 ctx), ((parallel_generatedRoot_251.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_251.outs ctx

theorem parallel_generatedRoot_251.facts_0  {ctx : parallel_generatedRoot_251Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_251.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_251.w_0_0 ctx) (parallel_generatedRoot_251.w_1_0 ctx) (parallel_generatedRoot_251.w_2_0 ctx) (parallel_generatedRoot_251.w_3_0 ctx) (parallel_generatedRoot_251.w_4_0 ctx) (parallel_generatedRoot_251.w_5_0 ctx) (parallel_generatedRoot_251.w_6_0 ctx) (parallel_generatedRoot_251.w_6_1 ctx) (parallel_generatedRoot_251.w_6_2 ctx) (parallel_generatedRoot_251.w_6_3 ctx) (parallel_generatedRoot_251.w_6_4 ctx) (parallel_generatedRoot_251.w_6_5 ctx) (parallel_generatedRoot_251.w_7_0 ctx) (parallel_generatedRoot_251.w_8_0 ctx) (parallel_generatedRoot_251.w_9_0 ctx) (parallel_generatedRoot_251.w_10_0 ctx) (parallel_generatedRoot_251.w_11_0 ctx) (parallel_generatedRoot_251.w_12_0 ctx) (parallel_generatedRoot_251.w_13_0 ctx) (parallel_generatedRoot_251.w_14_0 ctx) (parallel_generatedRoot_251.w_15_0 ctx) (parallel_generatedRoot_251.w_16_0 ctx) (parallel_generatedRoot_251.w_17_0 ctx) (parallel_generatedRoot_251.w_18_0 ctx) (parallel_generatedRoot_251.w_19_0 ctx) (parallel_generatedRoot_251.w_20_0 ctx) (parallel_generatedRoot_251.w_21_0 ctx) (parallel_generatedRoot_251.w_22_0 ctx) (parallel_generatedRoot_251.w_23_0 ctx) (parallel_generatedRoot_251.w_24_0 ctx) (parallel_generatedRoot_251.w_25_0 ctx) (parallel_generatedRoot_251.w_25_1 ctx) (parallel_generatedRoot_251.w_25_2 ctx) (parallel_generatedRoot_251.w_25_3 ctx) (parallel_generatedRoot_251.w_25_4 ctx) (parallel_generatedRoot_251.w_25_5 ctx) outputs) :
    parallel_generatedRoot_251.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_251.Facts  (ctx : parallel_generatedRoot_251Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_251.Facts_0 ctx outputs

theorem parallel_generatedRoot_251.facts  (ctx : parallel_generatedRoot_251Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_251.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_251.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_251.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_255.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_255.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_255Ctx" => (parallel_generatedRoot_255.Ctx backend hashModel params i_0)

def parallel_generatedRoot_255.w_0_0  (ctx : parallel_generatedRoot_255Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_255.w_1_0  (ctx : parallel_generatedRoot_255Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_255.w_2_0  (ctx : parallel_generatedRoot_255Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_255.w_3_0  (ctx : parallel_generatedRoot_255Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_255.w_4_0  (ctx : parallel_generatedRoot_255Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_255.w_5_0  (ctx : parallel_generatedRoot_255Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_255.w_6_0  (ctx : parallel_generatedRoot_255Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_255.w_6_1  (ctx : parallel_generatedRoot_255Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_255.w_6_2  (ctx : parallel_generatedRoot_255Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_255.w_6_3  (ctx : parallel_generatedRoot_255Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_255.w_6_4  (ctx : parallel_generatedRoot_255Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_255.w_6_5  (ctx : parallel_generatedRoot_255Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_255.w_7_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_255.w_8_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_255.w_9_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_255.w_7_0 ctx) (parallel_generatedRoot_255.w_8_0 ctx)

def parallel_generatedRoot_255.w_10_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_255.w_11_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_255.w_12_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_255.w_10_0 ctx) (parallel_generatedRoot_255.w_11_0 ctx)

def parallel_generatedRoot_255.w_13_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_255.w_14_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_255.w_15_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_255.w_13_0 ctx) (parallel_generatedRoot_255.w_14_0 ctx)

def parallel_generatedRoot_255.w_16_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_255.w_17_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_255.w_18_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_255.w_16_0 ctx) (parallel_generatedRoot_255.w_17_0 ctx)

def parallel_generatedRoot_255.w_19_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_255.w_20_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_255.w_21_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_255.w_19_0 ctx) (parallel_generatedRoot_255.w_20_0 ctx)

def parallel_generatedRoot_255.w_22_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_255.w_23_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_255.w_24_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_255.w_22_0 ctx) (parallel_generatedRoot_255.w_23_0 ctx)

def parallel_generatedRoot_255.w_25_0  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_255.w_25_1  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_255.w_25_2  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_255.w_25_3  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_255.w_25_4  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_255.w_25_5  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_255.outs  (ctx : parallel_generatedRoot_255Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_255.w_25_0 ctx), ((parallel_generatedRoot_255.w_25_1 ctx), ((parallel_generatedRoot_255.w_25_2 ctx), ((parallel_generatedRoot_255.w_25_3 ctx), ((parallel_generatedRoot_255.w_25_4 ctx), ((parallel_generatedRoot_255.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_255.Facts_0  (ctx : parallel_generatedRoot_255Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_255_6 backend hashModel params i_0 i ((parallel_generatedRoot_255.w_0_0 ctx), ((parallel_generatedRoot_255.w_1_0 ctx), ((parallel_generatedRoot_255.w_2_0 ctx), ((parallel_generatedRoot_255.w_3_0 ctx), ((parallel_generatedRoot_255.w_4_0 ctx), ((parallel_generatedRoot_255.w_5_0 ctx), ())))))) (((parallel_generatedRoot_255.w_6_0 ctx) i), (((parallel_generatedRoot_255.w_6_1 ctx) i), (((parallel_generatedRoot_255.w_6_2 ctx) i), (((parallel_generatedRoot_255.w_6_3 ctx) i), (((parallel_generatedRoot_255.w_6_4 ctx) i), (((parallel_generatedRoot_255.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_0 ctx) (0) (parallel_generatedRoot_255.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_0 ctx) (1) (parallel_generatedRoot_255.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_1 ctx) (0) (parallel_generatedRoot_255.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_1 ctx) (1) (parallel_generatedRoot_255.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_2 ctx) (0) (parallel_generatedRoot_255.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_2 ctx) (1) (parallel_generatedRoot_255.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_3 ctx) (0) (parallel_generatedRoot_255.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_3 ctx) (1) (parallel_generatedRoot_255.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_4 ctx) (0) (parallel_generatedRoot_255.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_4 ctx) (1) (parallel_generatedRoot_255.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_5 ctx) (0) (parallel_generatedRoot_255.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_255.w_6_5 ctx) (1) (parallel_generatedRoot_255.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_255_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_255.w_6_0 ctx), ((parallel_generatedRoot_255.w_6_1 ctx), ((parallel_generatedRoot_255.w_6_2 ctx), ((parallel_generatedRoot_255.w_6_3 ctx), ((parallel_generatedRoot_255.w_6_4 ctx), ((parallel_generatedRoot_255.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_255.w_9_0 ctx), ((parallel_generatedRoot_255.w_12_0 ctx), ((parallel_generatedRoot_255.w_15_0 ctx), ((parallel_generatedRoot_255.w_18_0 ctx), ((parallel_generatedRoot_255.w_21_0 ctx), ((parallel_generatedRoot_255.w_24_0 ctx), ())))))) ((parallel_generatedRoot_255.w_25_0 ctx), ((parallel_generatedRoot_255.w_25_1 ctx), ((parallel_generatedRoot_255.w_25_2 ctx), ((parallel_generatedRoot_255.w_25_3 ctx), ((parallel_generatedRoot_255.w_25_4 ctx), ((parallel_generatedRoot_255.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_255.outs ctx

theorem parallel_generatedRoot_255.facts_0  {ctx : parallel_generatedRoot_255Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_255.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_255.w_0_0 ctx) (parallel_generatedRoot_255.w_1_0 ctx) (parallel_generatedRoot_255.w_2_0 ctx) (parallel_generatedRoot_255.w_3_0 ctx) (parallel_generatedRoot_255.w_4_0 ctx) (parallel_generatedRoot_255.w_5_0 ctx) (parallel_generatedRoot_255.w_6_0 ctx) (parallel_generatedRoot_255.w_6_1 ctx) (parallel_generatedRoot_255.w_6_2 ctx) (parallel_generatedRoot_255.w_6_3 ctx) (parallel_generatedRoot_255.w_6_4 ctx) (parallel_generatedRoot_255.w_6_5 ctx) (parallel_generatedRoot_255.w_7_0 ctx) (parallel_generatedRoot_255.w_8_0 ctx) (parallel_generatedRoot_255.w_9_0 ctx) (parallel_generatedRoot_255.w_10_0 ctx) (parallel_generatedRoot_255.w_11_0 ctx) (parallel_generatedRoot_255.w_12_0 ctx) (parallel_generatedRoot_255.w_13_0 ctx) (parallel_generatedRoot_255.w_14_0 ctx) (parallel_generatedRoot_255.w_15_0 ctx) (parallel_generatedRoot_255.w_16_0 ctx) (parallel_generatedRoot_255.w_17_0 ctx) (parallel_generatedRoot_255.w_18_0 ctx) (parallel_generatedRoot_255.w_19_0 ctx) (parallel_generatedRoot_255.w_20_0 ctx) (parallel_generatedRoot_255.w_21_0 ctx) (parallel_generatedRoot_255.w_22_0 ctx) (parallel_generatedRoot_255.w_23_0 ctx) (parallel_generatedRoot_255.w_24_0 ctx) (parallel_generatedRoot_255.w_25_0 ctx) (parallel_generatedRoot_255.w_25_1 ctx) (parallel_generatedRoot_255.w_25_2 ctx) (parallel_generatedRoot_255.w_25_3 ctx) (parallel_generatedRoot_255.w_25_4 ctx) (parallel_generatedRoot_255.w_25_5 ctx) outputs) :
    parallel_generatedRoot_255.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_255.Facts  (ctx : parallel_generatedRoot_255Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_255.Facts_0 ctx outputs

theorem parallel_generatedRoot_255.facts  (ctx : parallel_generatedRoot_255Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_255.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_255.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_255.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_259.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_259.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_259Ctx" => (parallel_generatedRoot_259.Ctx backend hashModel params i_0)

def parallel_generatedRoot_259.w_0_0  (ctx : parallel_generatedRoot_259Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_259.w_1_0  (ctx : parallel_generatedRoot_259Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_259.w_2_0  (ctx : parallel_generatedRoot_259Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_259.w_3_0  (ctx : parallel_generatedRoot_259Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_259.w_4_0  (ctx : parallel_generatedRoot_259Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_259.w_5_0  (ctx : parallel_generatedRoot_259Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_259.w_6_0  (ctx : parallel_generatedRoot_259Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_259.w_6_1  (ctx : parallel_generatedRoot_259Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_259.w_6_2  (ctx : parallel_generatedRoot_259Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_259.w_6_3  (ctx : parallel_generatedRoot_259Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_259.w_6_4  (ctx : parallel_generatedRoot_259Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_259.w_6_5  (ctx : parallel_generatedRoot_259Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_259.w_7_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_259.w_8_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_259.w_9_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_259.w_7_0 ctx) (parallel_generatedRoot_259.w_8_0 ctx)

def parallel_generatedRoot_259.w_10_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_259.w_11_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_259.w_12_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_259.w_10_0 ctx) (parallel_generatedRoot_259.w_11_0 ctx)

def parallel_generatedRoot_259.w_13_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_259.w_14_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_259.w_15_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_259.w_13_0 ctx) (parallel_generatedRoot_259.w_14_0 ctx)

def parallel_generatedRoot_259.w_16_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_259.w_17_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_259.w_18_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_259.w_16_0 ctx) (parallel_generatedRoot_259.w_17_0 ctx)

def parallel_generatedRoot_259.w_19_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_259.w_20_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_259.w_21_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_259.w_19_0 ctx) (parallel_generatedRoot_259.w_20_0 ctx)

def parallel_generatedRoot_259.w_22_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_259.w_23_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_259.w_24_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_259.w_22_0 ctx) (parallel_generatedRoot_259.w_23_0 ctx)

def parallel_generatedRoot_259.w_25_0  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_259.w_25_1  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_259.w_25_2  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_259.w_25_3  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_259.w_25_4  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_259.w_25_5  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_259.outs  (ctx : parallel_generatedRoot_259Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_259.w_25_0 ctx), ((parallel_generatedRoot_259.w_25_1 ctx), ((parallel_generatedRoot_259.w_25_2 ctx), ((parallel_generatedRoot_259.w_25_3 ctx), ((parallel_generatedRoot_259.w_25_4 ctx), ((parallel_generatedRoot_259.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_259.Facts_0  (ctx : parallel_generatedRoot_259Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_259_6 backend hashModel params i_0 i ((parallel_generatedRoot_259.w_0_0 ctx), ((parallel_generatedRoot_259.w_1_0 ctx), ((parallel_generatedRoot_259.w_2_0 ctx), ((parallel_generatedRoot_259.w_3_0 ctx), ((parallel_generatedRoot_259.w_4_0 ctx), ((parallel_generatedRoot_259.w_5_0 ctx), ())))))) (((parallel_generatedRoot_259.w_6_0 ctx) i), (((parallel_generatedRoot_259.w_6_1 ctx) i), (((parallel_generatedRoot_259.w_6_2 ctx) i), (((parallel_generatedRoot_259.w_6_3 ctx) i), (((parallel_generatedRoot_259.w_6_4 ctx) i), (((parallel_generatedRoot_259.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_0 ctx) (0) (parallel_generatedRoot_259.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_0 ctx) (1) (parallel_generatedRoot_259.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_1 ctx) (0) (parallel_generatedRoot_259.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_1 ctx) (1) (parallel_generatedRoot_259.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_2 ctx) (0) (parallel_generatedRoot_259.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_2 ctx) (1) (parallel_generatedRoot_259.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_3 ctx) (0) (parallel_generatedRoot_259.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_3 ctx) (1) (parallel_generatedRoot_259.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_4 ctx) (0) (parallel_generatedRoot_259.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_4 ctx) (1) (parallel_generatedRoot_259.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_5 ctx) (0) (parallel_generatedRoot_259.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_259.w_6_5 ctx) (1) (parallel_generatedRoot_259.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_259_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_259.w_6_0 ctx), ((parallel_generatedRoot_259.w_6_1 ctx), ((parallel_generatedRoot_259.w_6_2 ctx), ((parallel_generatedRoot_259.w_6_3 ctx), ((parallel_generatedRoot_259.w_6_4 ctx), ((parallel_generatedRoot_259.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_259.w_9_0 ctx), ((parallel_generatedRoot_259.w_12_0 ctx), ((parallel_generatedRoot_259.w_15_0 ctx), ((parallel_generatedRoot_259.w_18_0 ctx), ((parallel_generatedRoot_259.w_21_0 ctx), ((parallel_generatedRoot_259.w_24_0 ctx), ())))))) ((parallel_generatedRoot_259.w_25_0 ctx), ((parallel_generatedRoot_259.w_25_1 ctx), ((parallel_generatedRoot_259.w_25_2 ctx), ((parallel_generatedRoot_259.w_25_3 ctx), ((parallel_generatedRoot_259.w_25_4 ctx), ((parallel_generatedRoot_259.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_259.outs ctx

theorem parallel_generatedRoot_259.facts_0  {ctx : parallel_generatedRoot_259Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_259.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_259.w_0_0 ctx) (parallel_generatedRoot_259.w_1_0 ctx) (parallel_generatedRoot_259.w_2_0 ctx) (parallel_generatedRoot_259.w_3_0 ctx) (parallel_generatedRoot_259.w_4_0 ctx) (parallel_generatedRoot_259.w_5_0 ctx) (parallel_generatedRoot_259.w_6_0 ctx) (parallel_generatedRoot_259.w_6_1 ctx) (parallel_generatedRoot_259.w_6_2 ctx) (parallel_generatedRoot_259.w_6_3 ctx) (parallel_generatedRoot_259.w_6_4 ctx) (parallel_generatedRoot_259.w_6_5 ctx) (parallel_generatedRoot_259.w_7_0 ctx) (parallel_generatedRoot_259.w_8_0 ctx) (parallel_generatedRoot_259.w_9_0 ctx) (parallel_generatedRoot_259.w_10_0 ctx) (parallel_generatedRoot_259.w_11_0 ctx) (parallel_generatedRoot_259.w_12_0 ctx) (parallel_generatedRoot_259.w_13_0 ctx) (parallel_generatedRoot_259.w_14_0 ctx) (parallel_generatedRoot_259.w_15_0 ctx) (parallel_generatedRoot_259.w_16_0 ctx) (parallel_generatedRoot_259.w_17_0 ctx) (parallel_generatedRoot_259.w_18_0 ctx) (parallel_generatedRoot_259.w_19_0 ctx) (parallel_generatedRoot_259.w_20_0 ctx) (parallel_generatedRoot_259.w_21_0 ctx) (parallel_generatedRoot_259.w_22_0 ctx) (parallel_generatedRoot_259.w_23_0 ctx) (parallel_generatedRoot_259.w_24_0 ctx) (parallel_generatedRoot_259.w_25_0 ctx) (parallel_generatedRoot_259.w_25_1 ctx) (parallel_generatedRoot_259.w_25_2 ctx) (parallel_generatedRoot_259.w_25_3 ctx) (parallel_generatedRoot_259.w_25_4 ctx) (parallel_generatedRoot_259.w_25_5 ctx) outputs) :
    parallel_generatedRoot_259.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_259.Facts  (ctx : parallel_generatedRoot_259Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_259.Facts_0 ctx outputs

theorem parallel_generatedRoot_259.facts  (ctx : parallel_generatedRoot_259Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_259.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_259.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_259.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_263.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_263.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_263Ctx" => (parallel_generatedRoot_263.Ctx backend hashModel params i_0)

def parallel_generatedRoot_263.w_0_0  (ctx : parallel_generatedRoot_263Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_263.w_1_0  (ctx : parallel_generatedRoot_263Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_263.w_2_0  (ctx : parallel_generatedRoot_263Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_263.w_3_0  (ctx : parallel_generatedRoot_263Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_263.w_4_0  (ctx : parallel_generatedRoot_263Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_263.w_5_0  (ctx : parallel_generatedRoot_263Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_263.w_6_0  (ctx : parallel_generatedRoot_263Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_263.w_6_1  (ctx : parallel_generatedRoot_263Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_263.w_6_2  (ctx : parallel_generatedRoot_263Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_263.w_6_3  (ctx : parallel_generatedRoot_263Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_263.w_6_4  (ctx : parallel_generatedRoot_263Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_263.w_6_5  (ctx : parallel_generatedRoot_263Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_263.w_7_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_263.w_8_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_263.w_9_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_263.w_7_0 ctx) (parallel_generatedRoot_263.w_8_0 ctx)

def parallel_generatedRoot_263.w_10_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_263.w_11_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_263.w_12_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_263.w_10_0 ctx) (parallel_generatedRoot_263.w_11_0 ctx)

def parallel_generatedRoot_263.w_13_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_263.w_14_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_263.w_15_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_263.w_13_0 ctx) (parallel_generatedRoot_263.w_14_0 ctx)

def parallel_generatedRoot_263.w_16_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_263.w_17_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_263.w_18_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_263.w_16_0 ctx) (parallel_generatedRoot_263.w_17_0 ctx)

def parallel_generatedRoot_263.w_19_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_263.w_20_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_263.w_21_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_263.w_19_0 ctx) (parallel_generatedRoot_263.w_20_0 ctx)

def parallel_generatedRoot_263.w_22_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_263.w_23_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_263.w_24_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_263.w_22_0 ctx) (parallel_generatedRoot_263.w_23_0 ctx)

def parallel_generatedRoot_263.w_25_0  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_263.w_25_1  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_263.w_25_2  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_263.w_25_3  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_263.w_25_4  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_263.w_25_5  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_263.outs  (ctx : parallel_generatedRoot_263Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_263.w_25_0 ctx), ((parallel_generatedRoot_263.w_25_1 ctx), ((parallel_generatedRoot_263.w_25_2 ctx), ((parallel_generatedRoot_263.w_25_3 ctx), ((parallel_generatedRoot_263.w_25_4 ctx), ((parallel_generatedRoot_263.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_263.Facts_0  (ctx : parallel_generatedRoot_263Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_263_6 backend hashModel params i_0 i ((parallel_generatedRoot_263.w_0_0 ctx), ((parallel_generatedRoot_263.w_1_0 ctx), ((parallel_generatedRoot_263.w_2_0 ctx), ((parallel_generatedRoot_263.w_3_0 ctx), ((parallel_generatedRoot_263.w_4_0 ctx), ((parallel_generatedRoot_263.w_5_0 ctx), ())))))) (((parallel_generatedRoot_263.w_6_0 ctx) i), (((parallel_generatedRoot_263.w_6_1 ctx) i), (((parallel_generatedRoot_263.w_6_2 ctx) i), (((parallel_generatedRoot_263.w_6_3 ctx) i), (((parallel_generatedRoot_263.w_6_4 ctx) i), (((parallel_generatedRoot_263.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_0 ctx) (0) (parallel_generatedRoot_263.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_0 ctx) (1) (parallel_generatedRoot_263.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_1 ctx) (0) (parallel_generatedRoot_263.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_1 ctx) (1) (parallel_generatedRoot_263.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_2 ctx) (0) (parallel_generatedRoot_263.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_2 ctx) (1) (parallel_generatedRoot_263.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_3 ctx) (0) (parallel_generatedRoot_263.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_3 ctx) (1) (parallel_generatedRoot_263.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_4 ctx) (0) (parallel_generatedRoot_263.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_4 ctx) (1) (parallel_generatedRoot_263.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_5 ctx) (0) (parallel_generatedRoot_263.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_263.w_6_5 ctx) (1) (parallel_generatedRoot_263.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_263_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_263.w_6_0 ctx), ((parallel_generatedRoot_263.w_6_1 ctx), ((parallel_generatedRoot_263.w_6_2 ctx), ((parallel_generatedRoot_263.w_6_3 ctx), ((parallel_generatedRoot_263.w_6_4 ctx), ((parallel_generatedRoot_263.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_263.w_9_0 ctx), ((parallel_generatedRoot_263.w_12_0 ctx), ((parallel_generatedRoot_263.w_15_0 ctx), ((parallel_generatedRoot_263.w_18_0 ctx), ((parallel_generatedRoot_263.w_21_0 ctx), ((parallel_generatedRoot_263.w_24_0 ctx), ())))))) ((parallel_generatedRoot_263.w_25_0 ctx), ((parallel_generatedRoot_263.w_25_1 ctx), ((parallel_generatedRoot_263.w_25_2 ctx), ((parallel_generatedRoot_263.w_25_3 ctx), ((parallel_generatedRoot_263.w_25_4 ctx), ((parallel_generatedRoot_263.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_263.outs ctx

theorem parallel_generatedRoot_263.facts_0  {ctx : parallel_generatedRoot_263Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_263.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_263.w_0_0 ctx) (parallel_generatedRoot_263.w_1_0 ctx) (parallel_generatedRoot_263.w_2_0 ctx) (parallel_generatedRoot_263.w_3_0 ctx) (parallel_generatedRoot_263.w_4_0 ctx) (parallel_generatedRoot_263.w_5_0 ctx) (parallel_generatedRoot_263.w_6_0 ctx) (parallel_generatedRoot_263.w_6_1 ctx) (parallel_generatedRoot_263.w_6_2 ctx) (parallel_generatedRoot_263.w_6_3 ctx) (parallel_generatedRoot_263.w_6_4 ctx) (parallel_generatedRoot_263.w_6_5 ctx) (parallel_generatedRoot_263.w_7_0 ctx) (parallel_generatedRoot_263.w_8_0 ctx) (parallel_generatedRoot_263.w_9_0 ctx) (parallel_generatedRoot_263.w_10_0 ctx) (parallel_generatedRoot_263.w_11_0 ctx) (parallel_generatedRoot_263.w_12_0 ctx) (parallel_generatedRoot_263.w_13_0 ctx) (parallel_generatedRoot_263.w_14_0 ctx) (parallel_generatedRoot_263.w_15_0 ctx) (parallel_generatedRoot_263.w_16_0 ctx) (parallel_generatedRoot_263.w_17_0 ctx) (parallel_generatedRoot_263.w_18_0 ctx) (parallel_generatedRoot_263.w_19_0 ctx) (parallel_generatedRoot_263.w_20_0 ctx) (parallel_generatedRoot_263.w_21_0 ctx) (parallel_generatedRoot_263.w_22_0 ctx) (parallel_generatedRoot_263.w_23_0 ctx) (parallel_generatedRoot_263.w_24_0 ctx) (parallel_generatedRoot_263.w_25_0 ctx) (parallel_generatedRoot_263.w_25_1 ctx) (parallel_generatedRoot_263.w_25_2 ctx) (parallel_generatedRoot_263.w_25_3 ctx) (parallel_generatedRoot_263.w_25_4 ctx) (parallel_generatedRoot_263.w_25_5 ctx) outputs) :
    parallel_generatedRoot_263.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_263.Facts  (ctx : parallel_generatedRoot_263Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_263.Facts_0 ctx outputs

theorem parallel_generatedRoot_263.facts  (ctx : parallel_generatedRoot_263Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_263.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_263.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_263.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_267.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_267.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_267Ctx" => (parallel_generatedRoot_267.Ctx backend hashModel params i_0)

def parallel_generatedRoot_267.w_0_0  (ctx : parallel_generatedRoot_267Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_267.w_1_0  (ctx : parallel_generatedRoot_267Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_267.w_2_0  (ctx : parallel_generatedRoot_267Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_267.w_3_0  (ctx : parallel_generatedRoot_267Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_267.w_4_0  (ctx : parallel_generatedRoot_267Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_267.w_5_0  (ctx : parallel_generatedRoot_267Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_267.w_6_0  (ctx : parallel_generatedRoot_267Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_267.w_6_1  (ctx : parallel_generatedRoot_267Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_267.w_6_2  (ctx : parallel_generatedRoot_267Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_267.w_6_3  (ctx : parallel_generatedRoot_267Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_267.w_6_4  (ctx : parallel_generatedRoot_267Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_267.w_6_5  (ctx : parallel_generatedRoot_267Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_267.w_7_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_267.w_8_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_267.w_9_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_267.w_7_0 ctx) (parallel_generatedRoot_267.w_8_0 ctx)

def parallel_generatedRoot_267.w_10_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_267.w_11_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_267.w_12_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_267.w_10_0 ctx) (parallel_generatedRoot_267.w_11_0 ctx)

def parallel_generatedRoot_267.w_13_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_267.w_14_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_267.w_15_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_267.w_13_0 ctx) (parallel_generatedRoot_267.w_14_0 ctx)

def parallel_generatedRoot_267.w_16_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_267.w_17_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_267.w_18_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_267.w_16_0 ctx) (parallel_generatedRoot_267.w_17_0 ctx)

def parallel_generatedRoot_267.w_19_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_267.w_20_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_267.w_21_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_267.w_19_0 ctx) (parallel_generatedRoot_267.w_20_0 ctx)

def parallel_generatedRoot_267.w_22_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_267.w_23_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_267.w_24_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_267.w_22_0 ctx) (parallel_generatedRoot_267.w_23_0 ctx)

def parallel_generatedRoot_267.w_25_0  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_267.w_25_1  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_267.w_25_2  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_267.w_25_3  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_267.w_25_4  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_267.w_25_5  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_267.outs  (ctx : parallel_generatedRoot_267Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_267.w_25_0 ctx), ((parallel_generatedRoot_267.w_25_1 ctx), ((parallel_generatedRoot_267.w_25_2 ctx), ((parallel_generatedRoot_267.w_25_3 ctx), ((parallel_generatedRoot_267.w_25_4 ctx), ((parallel_generatedRoot_267.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_267.Facts_0  (ctx : parallel_generatedRoot_267Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_267_6 backend hashModel params i_0 i ((parallel_generatedRoot_267.w_0_0 ctx), ((parallel_generatedRoot_267.w_1_0 ctx), ((parallel_generatedRoot_267.w_2_0 ctx), ((parallel_generatedRoot_267.w_3_0 ctx), ((parallel_generatedRoot_267.w_4_0 ctx), ((parallel_generatedRoot_267.w_5_0 ctx), ())))))) (((parallel_generatedRoot_267.w_6_0 ctx) i), (((parallel_generatedRoot_267.w_6_1 ctx) i), (((parallel_generatedRoot_267.w_6_2 ctx) i), (((parallel_generatedRoot_267.w_6_3 ctx) i), (((parallel_generatedRoot_267.w_6_4 ctx) i), (((parallel_generatedRoot_267.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_0 ctx) (0) (parallel_generatedRoot_267.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_0 ctx) (1) (parallel_generatedRoot_267.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_1 ctx) (0) (parallel_generatedRoot_267.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_1 ctx) (1) (parallel_generatedRoot_267.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_2 ctx) (0) (parallel_generatedRoot_267.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_2 ctx) (1) (parallel_generatedRoot_267.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_3 ctx) (0) (parallel_generatedRoot_267.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_3 ctx) (1) (parallel_generatedRoot_267.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_4 ctx) (0) (parallel_generatedRoot_267.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_4 ctx) (1) (parallel_generatedRoot_267.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_5 ctx) (0) (parallel_generatedRoot_267.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_267.w_6_5 ctx) (1) (parallel_generatedRoot_267.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_267_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_267.w_6_0 ctx), ((parallel_generatedRoot_267.w_6_1 ctx), ((parallel_generatedRoot_267.w_6_2 ctx), ((parallel_generatedRoot_267.w_6_3 ctx), ((parallel_generatedRoot_267.w_6_4 ctx), ((parallel_generatedRoot_267.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_267.w_9_0 ctx), ((parallel_generatedRoot_267.w_12_0 ctx), ((parallel_generatedRoot_267.w_15_0 ctx), ((parallel_generatedRoot_267.w_18_0 ctx), ((parallel_generatedRoot_267.w_21_0 ctx), ((parallel_generatedRoot_267.w_24_0 ctx), ())))))) ((parallel_generatedRoot_267.w_25_0 ctx), ((parallel_generatedRoot_267.w_25_1 ctx), ((parallel_generatedRoot_267.w_25_2 ctx), ((parallel_generatedRoot_267.w_25_3 ctx), ((parallel_generatedRoot_267.w_25_4 ctx), ((parallel_generatedRoot_267.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_267.outs ctx

theorem parallel_generatedRoot_267.facts_0  {ctx : parallel_generatedRoot_267Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_267.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_267.w_0_0 ctx) (parallel_generatedRoot_267.w_1_0 ctx) (parallel_generatedRoot_267.w_2_0 ctx) (parallel_generatedRoot_267.w_3_0 ctx) (parallel_generatedRoot_267.w_4_0 ctx) (parallel_generatedRoot_267.w_5_0 ctx) (parallel_generatedRoot_267.w_6_0 ctx) (parallel_generatedRoot_267.w_6_1 ctx) (parallel_generatedRoot_267.w_6_2 ctx) (parallel_generatedRoot_267.w_6_3 ctx) (parallel_generatedRoot_267.w_6_4 ctx) (parallel_generatedRoot_267.w_6_5 ctx) (parallel_generatedRoot_267.w_7_0 ctx) (parallel_generatedRoot_267.w_8_0 ctx) (parallel_generatedRoot_267.w_9_0 ctx) (parallel_generatedRoot_267.w_10_0 ctx) (parallel_generatedRoot_267.w_11_0 ctx) (parallel_generatedRoot_267.w_12_0 ctx) (parallel_generatedRoot_267.w_13_0 ctx) (parallel_generatedRoot_267.w_14_0 ctx) (parallel_generatedRoot_267.w_15_0 ctx) (parallel_generatedRoot_267.w_16_0 ctx) (parallel_generatedRoot_267.w_17_0 ctx) (parallel_generatedRoot_267.w_18_0 ctx) (parallel_generatedRoot_267.w_19_0 ctx) (parallel_generatedRoot_267.w_20_0 ctx) (parallel_generatedRoot_267.w_21_0 ctx) (parallel_generatedRoot_267.w_22_0 ctx) (parallel_generatedRoot_267.w_23_0 ctx) (parallel_generatedRoot_267.w_24_0 ctx) (parallel_generatedRoot_267.w_25_0 ctx) (parallel_generatedRoot_267.w_25_1 ctx) (parallel_generatedRoot_267.w_25_2 ctx) (parallel_generatedRoot_267.w_25_3 ctx) (parallel_generatedRoot_267.w_25_4 ctx) (parallel_generatedRoot_267.w_25_5 ctx) outputs) :
    parallel_generatedRoot_267.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_267.Facts  (ctx : parallel_generatedRoot_267Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_267.Facts_0 ctx outputs

theorem parallel_generatedRoot_267.facts  (ctx : parallel_generatedRoot_267Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_267.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_267.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_267.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_271.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_271.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_271Ctx" => (parallel_generatedRoot_271.Ctx backend hashModel params i_0)

def parallel_generatedRoot_271.w_0_0  (ctx : parallel_generatedRoot_271Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_271.w_1_0  (ctx : parallel_generatedRoot_271Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_271.w_2_0  (ctx : parallel_generatedRoot_271Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_271.w_3_0  (ctx : parallel_generatedRoot_271Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_271.w_4_0  (ctx : parallel_generatedRoot_271Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_271.w_5_0  (ctx : parallel_generatedRoot_271Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_271.w_6_0  (ctx : parallel_generatedRoot_271Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_271.w_6_1  (ctx : parallel_generatedRoot_271Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_271.w_6_2  (ctx : parallel_generatedRoot_271Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_271.w_6_3  (ctx : parallel_generatedRoot_271Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_271.w_6_4  (ctx : parallel_generatedRoot_271Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_271.w_6_5  (ctx : parallel_generatedRoot_271Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_271.w_7_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_271.w_8_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_271.w_9_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_271.w_7_0 ctx) (parallel_generatedRoot_271.w_8_0 ctx)

def parallel_generatedRoot_271.w_10_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_271.w_11_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_271.w_12_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_271.w_10_0 ctx) (parallel_generatedRoot_271.w_11_0 ctx)

def parallel_generatedRoot_271.w_13_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_271.w_14_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_271.w_15_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_271.w_13_0 ctx) (parallel_generatedRoot_271.w_14_0 ctx)

def parallel_generatedRoot_271.w_16_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_271.w_17_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_271.w_18_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_271.w_16_0 ctx) (parallel_generatedRoot_271.w_17_0 ctx)

def parallel_generatedRoot_271.w_19_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_271.w_20_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_271.w_21_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_271.w_19_0 ctx) (parallel_generatedRoot_271.w_20_0 ctx)

def parallel_generatedRoot_271.w_22_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_271.w_23_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_271.w_24_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_271.w_22_0 ctx) (parallel_generatedRoot_271.w_23_0 ctx)

def parallel_generatedRoot_271.w_25_0  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_271.w_25_1  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_271.w_25_2  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_271.w_25_3  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_271.w_25_4  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_271.w_25_5  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_271.outs  (ctx : parallel_generatedRoot_271Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_271.w_25_0 ctx), ((parallel_generatedRoot_271.w_25_1 ctx), ((parallel_generatedRoot_271.w_25_2 ctx), ((parallel_generatedRoot_271.w_25_3 ctx), ((parallel_generatedRoot_271.w_25_4 ctx), ((parallel_generatedRoot_271.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_271.Facts_0  (ctx : parallel_generatedRoot_271Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_271_6 backend hashModel params i_0 i ((parallel_generatedRoot_271.w_0_0 ctx), ((parallel_generatedRoot_271.w_1_0 ctx), ((parallel_generatedRoot_271.w_2_0 ctx), ((parallel_generatedRoot_271.w_3_0 ctx), ((parallel_generatedRoot_271.w_4_0 ctx), ((parallel_generatedRoot_271.w_5_0 ctx), ())))))) (((parallel_generatedRoot_271.w_6_0 ctx) i), (((parallel_generatedRoot_271.w_6_1 ctx) i), (((parallel_generatedRoot_271.w_6_2 ctx) i), (((parallel_generatedRoot_271.w_6_3 ctx) i), (((parallel_generatedRoot_271.w_6_4 ctx) i), (((parallel_generatedRoot_271.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_0 ctx) (0) (parallel_generatedRoot_271.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_0 ctx) (1) (parallel_generatedRoot_271.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_1 ctx) (0) (parallel_generatedRoot_271.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_1 ctx) (1) (parallel_generatedRoot_271.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_2 ctx) (0) (parallel_generatedRoot_271.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_2 ctx) (1) (parallel_generatedRoot_271.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_3 ctx) (0) (parallel_generatedRoot_271.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_3 ctx) (1) (parallel_generatedRoot_271.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_4 ctx) (0) (parallel_generatedRoot_271.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_4 ctx) (1) (parallel_generatedRoot_271.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_5 ctx) (0) (parallel_generatedRoot_271.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_271.w_6_5 ctx) (1) (parallel_generatedRoot_271.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_271_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_271.w_6_0 ctx), ((parallel_generatedRoot_271.w_6_1 ctx), ((parallel_generatedRoot_271.w_6_2 ctx), ((parallel_generatedRoot_271.w_6_3 ctx), ((parallel_generatedRoot_271.w_6_4 ctx), ((parallel_generatedRoot_271.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_271.w_9_0 ctx), ((parallel_generatedRoot_271.w_12_0 ctx), ((parallel_generatedRoot_271.w_15_0 ctx), ((parallel_generatedRoot_271.w_18_0 ctx), ((parallel_generatedRoot_271.w_21_0 ctx), ((parallel_generatedRoot_271.w_24_0 ctx), ())))))) ((parallel_generatedRoot_271.w_25_0 ctx), ((parallel_generatedRoot_271.w_25_1 ctx), ((parallel_generatedRoot_271.w_25_2 ctx), ((parallel_generatedRoot_271.w_25_3 ctx), ((parallel_generatedRoot_271.w_25_4 ctx), ((parallel_generatedRoot_271.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_271.outs ctx

theorem parallel_generatedRoot_271.facts_0  {ctx : parallel_generatedRoot_271Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_271.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_271.w_0_0 ctx) (parallel_generatedRoot_271.w_1_0 ctx) (parallel_generatedRoot_271.w_2_0 ctx) (parallel_generatedRoot_271.w_3_0 ctx) (parallel_generatedRoot_271.w_4_0 ctx) (parallel_generatedRoot_271.w_5_0 ctx) (parallel_generatedRoot_271.w_6_0 ctx) (parallel_generatedRoot_271.w_6_1 ctx) (parallel_generatedRoot_271.w_6_2 ctx) (parallel_generatedRoot_271.w_6_3 ctx) (parallel_generatedRoot_271.w_6_4 ctx) (parallel_generatedRoot_271.w_6_5 ctx) (parallel_generatedRoot_271.w_7_0 ctx) (parallel_generatedRoot_271.w_8_0 ctx) (parallel_generatedRoot_271.w_9_0 ctx) (parallel_generatedRoot_271.w_10_0 ctx) (parallel_generatedRoot_271.w_11_0 ctx) (parallel_generatedRoot_271.w_12_0 ctx) (parallel_generatedRoot_271.w_13_0 ctx) (parallel_generatedRoot_271.w_14_0 ctx) (parallel_generatedRoot_271.w_15_0 ctx) (parallel_generatedRoot_271.w_16_0 ctx) (parallel_generatedRoot_271.w_17_0 ctx) (parallel_generatedRoot_271.w_18_0 ctx) (parallel_generatedRoot_271.w_19_0 ctx) (parallel_generatedRoot_271.w_20_0 ctx) (parallel_generatedRoot_271.w_21_0 ctx) (parallel_generatedRoot_271.w_22_0 ctx) (parallel_generatedRoot_271.w_23_0 ctx) (parallel_generatedRoot_271.w_24_0 ctx) (parallel_generatedRoot_271.w_25_0 ctx) (parallel_generatedRoot_271.w_25_1 ctx) (parallel_generatedRoot_271.w_25_2 ctx) (parallel_generatedRoot_271.w_25_3 ctx) (parallel_generatedRoot_271.w_25_4 ctx) (parallel_generatedRoot_271.w_25_5 ctx) outputs) :
    parallel_generatedRoot_271.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_271.Facts  (ctx : parallel_generatedRoot_271Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_271.Facts_0 ctx outputs

theorem parallel_generatedRoot_271.facts  (ctx : parallel_generatedRoot_271Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_271.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_271.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_271.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_275.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_275.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_275Ctx" => (parallel_generatedRoot_275.Ctx backend hashModel params i_0)

def parallel_generatedRoot_275.w_0_0  (ctx : parallel_generatedRoot_275Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_275.w_1_0  (ctx : parallel_generatedRoot_275Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_275.w_2_0  (ctx : parallel_generatedRoot_275Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_275.w_3_0  (ctx : parallel_generatedRoot_275Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_275.w_4_0  (ctx : parallel_generatedRoot_275Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_275.w_5_0  (ctx : parallel_generatedRoot_275Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_275.w_6_0  (ctx : parallel_generatedRoot_275Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_275.w_6_1  (ctx : parallel_generatedRoot_275Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_275.w_6_2  (ctx : parallel_generatedRoot_275Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_275.w_6_3  (ctx : parallel_generatedRoot_275Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_275.w_6_4  (ctx : parallel_generatedRoot_275Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_275.w_6_5  (ctx : parallel_generatedRoot_275Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_275.w_7_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_275.w_8_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_275.w_9_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_275.w_7_0 ctx) (parallel_generatedRoot_275.w_8_0 ctx)

def parallel_generatedRoot_275.w_10_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_275.w_11_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_275.w_12_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_275.w_10_0 ctx) (parallel_generatedRoot_275.w_11_0 ctx)

def parallel_generatedRoot_275.w_13_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_275.w_14_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_275.w_15_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_275.w_13_0 ctx) (parallel_generatedRoot_275.w_14_0 ctx)

def parallel_generatedRoot_275.w_16_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_275.w_17_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_275.w_18_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_275.w_16_0 ctx) (parallel_generatedRoot_275.w_17_0 ctx)

def parallel_generatedRoot_275.w_19_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_275.w_20_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_275.w_21_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_275.w_19_0 ctx) (parallel_generatedRoot_275.w_20_0 ctx)

def parallel_generatedRoot_275.w_22_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_275.w_23_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_275.w_24_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_275.w_22_0 ctx) (parallel_generatedRoot_275.w_23_0 ctx)

def parallel_generatedRoot_275.w_25_0  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_275.w_25_1  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_275.w_25_2  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_275.w_25_3  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_275.w_25_4  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_275.w_25_5  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_275.outs  (ctx : parallel_generatedRoot_275Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_275.w_25_0 ctx), ((parallel_generatedRoot_275.w_25_1 ctx), ((parallel_generatedRoot_275.w_25_2 ctx), ((parallel_generatedRoot_275.w_25_3 ctx), ((parallel_generatedRoot_275.w_25_4 ctx), ((parallel_generatedRoot_275.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_275.Facts_0  (ctx : parallel_generatedRoot_275Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_275_6 backend hashModel params i_0 i ((parallel_generatedRoot_275.w_0_0 ctx), ((parallel_generatedRoot_275.w_1_0 ctx), ((parallel_generatedRoot_275.w_2_0 ctx), ((parallel_generatedRoot_275.w_3_0 ctx), ((parallel_generatedRoot_275.w_4_0 ctx), ((parallel_generatedRoot_275.w_5_0 ctx), ())))))) (((parallel_generatedRoot_275.w_6_0 ctx) i), (((parallel_generatedRoot_275.w_6_1 ctx) i), (((parallel_generatedRoot_275.w_6_2 ctx) i), (((parallel_generatedRoot_275.w_6_3 ctx) i), (((parallel_generatedRoot_275.w_6_4 ctx) i), (((parallel_generatedRoot_275.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_0 ctx) (0) (parallel_generatedRoot_275.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_0 ctx) (1) (parallel_generatedRoot_275.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_1 ctx) (0) (parallel_generatedRoot_275.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_1 ctx) (1) (parallel_generatedRoot_275.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_2 ctx) (0) (parallel_generatedRoot_275.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_2 ctx) (1) (parallel_generatedRoot_275.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_3 ctx) (0) (parallel_generatedRoot_275.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_3 ctx) (1) (parallel_generatedRoot_275.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_4 ctx) (0) (parallel_generatedRoot_275.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_4 ctx) (1) (parallel_generatedRoot_275.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_5 ctx) (0) (parallel_generatedRoot_275.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_275.w_6_5 ctx) (1) (parallel_generatedRoot_275.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_275_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_275.w_6_0 ctx), ((parallel_generatedRoot_275.w_6_1 ctx), ((parallel_generatedRoot_275.w_6_2 ctx), ((parallel_generatedRoot_275.w_6_3 ctx), ((parallel_generatedRoot_275.w_6_4 ctx), ((parallel_generatedRoot_275.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_275.w_9_0 ctx), ((parallel_generatedRoot_275.w_12_0 ctx), ((parallel_generatedRoot_275.w_15_0 ctx), ((parallel_generatedRoot_275.w_18_0 ctx), ((parallel_generatedRoot_275.w_21_0 ctx), ((parallel_generatedRoot_275.w_24_0 ctx), ())))))) ((parallel_generatedRoot_275.w_25_0 ctx), ((parallel_generatedRoot_275.w_25_1 ctx), ((parallel_generatedRoot_275.w_25_2 ctx), ((parallel_generatedRoot_275.w_25_3 ctx), ((parallel_generatedRoot_275.w_25_4 ctx), ((parallel_generatedRoot_275.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_275.outs ctx

theorem parallel_generatedRoot_275.facts_0  {ctx : parallel_generatedRoot_275Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_275.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_275.w_0_0 ctx) (parallel_generatedRoot_275.w_1_0 ctx) (parallel_generatedRoot_275.w_2_0 ctx) (parallel_generatedRoot_275.w_3_0 ctx) (parallel_generatedRoot_275.w_4_0 ctx) (parallel_generatedRoot_275.w_5_0 ctx) (parallel_generatedRoot_275.w_6_0 ctx) (parallel_generatedRoot_275.w_6_1 ctx) (parallel_generatedRoot_275.w_6_2 ctx) (parallel_generatedRoot_275.w_6_3 ctx) (parallel_generatedRoot_275.w_6_4 ctx) (parallel_generatedRoot_275.w_6_5 ctx) (parallel_generatedRoot_275.w_7_0 ctx) (parallel_generatedRoot_275.w_8_0 ctx) (parallel_generatedRoot_275.w_9_0 ctx) (parallel_generatedRoot_275.w_10_0 ctx) (parallel_generatedRoot_275.w_11_0 ctx) (parallel_generatedRoot_275.w_12_0 ctx) (parallel_generatedRoot_275.w_13_0 ctx) (parallel_generatedRoot_275.w_14_0 ctx) (parallel_generatedRoot_275.w_15_0 ctx) (parallel_generatedRoot_275.w_16_0 ctx) (parallel_generatedRoot_275.w_17_0 ctx) (parallel_generatedRoot_275.w_18_0 ctx) (parallel_generatedRoot_275.w_19_0 ctx) (parallel_generatedRoot_275.w_20_0 ctx) (parallel_generatedRoot_275.w_21_0 ctx) (parallel_generatedRoot_275.w_22_0 ctx) (parallel_generatedRoot_275.w_23_0 ctx) (parallel_generatedRoot_275.w_24_0 ctx) (parallel_generatedRoot_275.w_25_0 ctx) (parallel_generatedRoot_275.w_25_1 ctx) (parallel_generatedRoot_275.w_25_2 ctx) (parallel_generatedRoot_275.w_25_3 ctx) (parallel_generatedRoot_275.w_25_4 ctx) (parallel_generatedRoot_275.w_25_5 ctx) outputs) :
    parallel_generatedRoot_275.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_275.Facts  (ctx : parallel_generatedRoot_275Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_275.Facts_0 ctx outputs

theorem parallel_generatedRoot_275.facts  (ctx : parallel_generatedRoot_275Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_275.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_275.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_275.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_279.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_279.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_279Ctx" => (parallel_generatedRoot_279.Ctx backend hashModel params i_0)

def parallel_generatedRoot_279.w_0_0  (ctx : parallel_generatedRoot_279Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_279.w_1_0  (ctx : parallel_generatedRoot_279Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_279.w_2_0  (ctx : parallel_generatedRoot_279Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_279.w_3_0  (ctx : parallel_generatedRoot_279Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_279.w_4_0  (ctx : parallel_generatedRoot_279Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_279.w_5_0  (ctx : parallel_generatedRoot_279Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_279.w_6_0  (ctx : parallel_generatedRoot_279Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_279.w_6_1  (ctx : parallel_generatedRoot_279Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_279.w_6_2  (ctx : parallel_generatedRoot_279Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_279.w_6_3  (ctx : parallel_generatedRoot_279Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_279.w_6_4  (ctx : parallel_generatedRoot_279Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_279.w_6_5  (ctx : parallel_generatedRoot_279Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_279.w_7_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_279.w_8_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_279.w_9_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_279.w_7_0 ctx) (parallel_generatedRoot_279.w_8_0 ctx)

def parallel_generatedRoot_279.w_10_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_279.w_11_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_279.w_12_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_279.w_10_0 ctx) (parallel_generatedRoot_279.w_11_0 ctx)

def parallel_generatedRoot_279.w_13_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_279.w_14_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_279.w_15_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_279.w_13_0 ctx) (parallel_generatedRoot_279.w_14_0 ctx)

def parallel_generatedRoot_279.w_16_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_279.w_17_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_279.w_18_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_279.w_16_0 ctx) (parallel_generatedRoot_279.w_17_0 ctx)

def parallel_generatedRoot_279.w_19_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_279.w_20_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_279.w_21_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_279.w_19_0 ctx) (parallel_generatedRoot_279.w_20_0 ctx)

def parallel_generatedRoot_279.w_22_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_279.w_23_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_279.w_24_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_279.w_22_0 ctx) (parallel_generatedRoot_279.w_23_0 ctx)

def parallel_generatedRoot_279.w_25_0  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_279.w_25_1  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_279.w_25_2  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_279.w_25_3  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_279.w_25_4  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_279.w_25_5  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_279.outs  (ctx : parallel_generatedRoot_279Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_279.w_25_0 ctx), ((parallel_generatedRoot_279.w_25_1 ctx), ((parallel_generatedRoot_279.w_25_2 ctx), ((parallel_generatedRoot_279.w_25_3 ctx), ((parallel_generatedRoot_279.w_25_4 ctx), ((parallel_generatedRoot_279.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_279.Facts_0  (ctx : parallel_generatedRoot_279Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_279_6 backend hashModel params i_0 i ((parallel_generatedRoot_279.w_0_0 ctx), ((parallel_generatedRoot_279.w_1_0 ctx), ((parallel_generatedRoot_279.w_2_0 ctx), ((parallel_generatedRoot_279.w_3_0 ctx), ((parallel_generatedRoot_279.w_4_0 ctx), ((parallel_generatedRoot_279.w_5_0 ctx), ())))))) (((parallel_generatedRoot_279.w_6_0 ctx) i), (((parallel_generatedRoot_279.w_6_1 ctx) i), (((parallel_generatedRoot_279.w_6_2 ctx) i), (((parallel_generatedRoot_279.w_6_3 ctx) i), (((parallel_generatedRoot_279.w_6_4 ctx) i), (((parallel_generatedRoot_279.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_0 ctx) (0) (parallel_generatedRoot_279.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_0 ctx) (1) (parallel_generatedRoot_279.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_1 ctx) (0) (parallel_generatedRoot_279.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_1 ctx) (1) (parallel_generatedRoot_279.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_2 ctx) (0) (parallel_generatedRoot_279.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_2 ctx) (1) (parallel_generatedRoot_279.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_3 ctx) (0) (parallel_generatedRoot_279.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_3 ctx) (1) (parallel_generatedRoot_279.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_4 ctx) (0) (parallel_generatedRoot_279.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_4 ctx) (1) (parallel_generatedRoot_279.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_5 ctx) (0) (parallel_generatedRoot_279.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_279.w_6_5 ctx) (1) (parallel_generatedRoot_279.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_279_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_279.w_6_0 ctx), ((parallel_generatedRoot_279.w_6_1 ctx), ((parallel_generatedRoot_279.w_6_2 ctx), ((parallel_generatedRoot_279.w_6_3 ctx), ((parallel_generatedRoot_279.w_6_4 ctx), ((parallel_generatedRoot_279.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_279.w_9_0 ctx), ((parallel_generatedRoot_279.w_12_0 ctx), ((parallel_generatedRoot_279.w_15_0 ctx), ((parallel_generatedRoot_279.w_18_0 ctx), ((parallel_generatedRoot_279.w_21_0 ctx), ((parallel_generatedRoot_279.w_24_0 ctx), ())))))) ((parallel_generatedRoot_279.w_25_0 ctx), ((parallel_generatedRoot_279.w_25_1 ctx), ((parallel_generatedRoot_279.w_25_2 ctx), ((parallel_generatedRoot_279.w_25_3 ctx), ((parallel_generatedRoot_279.w_25_4 ctx), ((parallel_generatedRoot_279.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_279.outs ctx

theorem parallel_generatedRoot_279.facts_0  {ctx : parallel_generatedRoot_279Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_279.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_279.w_0_0 ctx) (parallel_generatedRoot_279.w_1_0 ctx) (parallel_generatedRoot_279.w_2_0 ctx) (parallel_generatedRoot_279.w_3_0 ctx) (parallel_generatedRoot_279.w_4_0 ctx) (parallel_generatedRoot_279.w_5_0 ctx) (parallel_generatedRoot_279.w_6_0 ctx) (parallel_generatedRoot_279.w_6_1 ctx) (parallel_generatedRoot_279.w_6_2 ctx) (parallel_generatedRoot_279.w_6_3 ctx) (parallel_generatedRoot_279.w_6_4 ctx) (parallel_generatedRoot_279.w_6_5 ctx) (parallel_generatedRoot_279.w_7_0 ctx) (parallel_generatedRoot_279.w_8_0 ctx) (parallel_generatedRoot_279.w_9_0 ctx) (parallel_generatedRoot_279.w_10_0 ctx) (parallel_generatedRoot_279.w_11_0 ctx) (parallel_generatedRoot_279.w_12_0 ctx) (parallel_generatedRoot_279.w_13_0 ctx) (parallel_generatedRoot_279.w_14_0 ctx) (parallel_generatedRoot_279.w_15_0 ctx) (parallel_generatedRoot_279.w_16_0 ctx) (parallel_generatedRoot_279.w_17_0 ctx) (parallel_generatedRoot_279.w_18_0 ctx) (parallel_generatedRoot_279.w_19_0 ctx) (parallel_generatedRoot_279.w_20_0 ctx) (parallel_generatedRoot_279.w_21_0 ctx) (parallel_generatedRoot_279.w_22_0 ctx) (parallel_generatedRoot_279.w_23_0 ctx) (parallel_generatedRoot_279.w_24_0 ctx) (parallel_generatedRoot_279.w_25_0 ctx) (parallel_generatedRoot_279.w_25_1 ctx) (parallel_generatedRoot_279.w_25_2 ctx) (parallel_generatedRoot_279.w_25_3 ctx) (parallel_generatedRoot_279.w_25_4 ctx) (parallel_generatedRoot_279.w_25_5 ctx) outputs) :
    parallel_generatedRoot_279.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_279.Facts  (ctx : parallel_generatedRoot_279Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_279.Facts_0 ctx outputs

theorem parallel_generatedRoot_279.facts  (ctx : parallel_generatedRoot_279Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_279.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_279.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_279.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_283.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_283.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_283Ctx" => (parallel_generatedRoot_283.Ctx backend hashModel params i_0)

def parallel_generatedRoot_283.w_0_0  (ctx : parallel_generatedRoot_283Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_283.w_1_0  (ctx : parallel_generatedRoot_283Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_283.w_2_0  (ctx : parallel_generatedRoot_283Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_283.w_3_0  (ctx : parallel_generatedRoot_283Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_283.w_4_0  (ctx : parallel_generatedRoot_283Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_283.w_5_0  (ctx : parallel_generatedRoot_283Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_283.w_6_0  (ctx : parallel_generatedRoot_283Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_283.w_6_1  (ctx : parallel_generatedRoot_283Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_283.w_6_2  (ctx : parallel_generatedRoot_283Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_283.w_6_3  (ctx : parallel_generatedRoot_283Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_283.w_6_4  (ctx : parallel_generatedRoot_283Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_283.w_6_5  (ctx : parallel_generatedRoot_283Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_283.w_7_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_283.w_8_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_283.w_9_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_283.w_7_0 ctx) (parallel_generatedRoot_283.w_8_0 ctx)

def parallel_generatedRoot_283.w_10_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_283.w_11_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_283.w_12_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_283.w_10_0 ctx) (parallel_generatedRoot_283.w_11_0 ctx)

def parallel_generatedRoot_283.w_13_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_283.w_14_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_283.w_15_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_283.w_13_0 ctx) (parallel_generatedRoot_283.w_14_0 ctx)

def parallel_generatedRoot_283.w_16_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_283.w_17_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_283.w_18_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_283.w_16_0 ctx) (parallel_generatedRoot_283.w_17_0 ctx)

def parallel_generatedRoot_283.w_19_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_283.w_20_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_283.w_21_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_283.w_19_0 ctx) (parallel_generatedRoot_283.w_20_0 ctx)

def parallel_generatedRoot_283.w_22_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_283.w_23_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_283.w_24_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_283.w_22_0 ctx) (parallel_generatedRoot_283.w_23_0 ctx)

def parallel_generatedRoot_283.w_25_0  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_283.w_25_1  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_283.w_25_2  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_283.w_25_3  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_283.w_25_4  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_283.w_25_5  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_283.outs  (ctx : parallel_generatedRoot_283Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_283.w_25_0 ctx), ((parallel_generatedRoot_283.w_25_1 ctx), ((parallel_generatedRoot_283.w_25_2 ctx), ((parallel_generatedRoot_283.w_25_3 ctx), ((parallel_generatedRoot_283.w_25_4 ctx), ((parallel_generatedRoot_283.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_283.Facts_0  (ctx : parallel_generatedRoot_283Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_283_6 backend hashModel params i_0 i ((parallel_generatedRoot_283.w_0_0 ctx), ((parallel_generatedRoot_283.w_1_0 ctx), ((parallel_generatedRoot_283.w_2_0 ctx), ((parallel_generatedRoot_283.w_3_0 ctx), ((parallel_generatedRoot_283.w_4_0 ctx), ((parallel_generatedRoot_283.w_5_0 ctx), ())))))) (((parallel_generatedRoot_283.w_6_0 ctx) i), (((parallel_generatedRoot_283.w_6_1 ctx) i), (((parallel_generatedRoot_283.w_6_2 ctx) i), (((parallel_generatedRoot_283.w_6_3 ctx) i), (((parallel_generatedRoot_283.w_6_4 ctx) i), (((parallel_generatedRoot_283.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_0 ctx) (0) (parallel_generatedRoot_283.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_0 ctx) (1) (parallel_generatedRoot_283.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_1 ctx) (0) (parallel_generatedRoot_283.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_1 ctx) (1) (parallel_generatedRoot_283.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_2 ctx) (0) (parallel_generatedRoot_283.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_2 ctx) (1) (parallel_generatedRoot_283.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_3 ctx) (0) (parallel_generatedRoot_283.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_3 ctx) (1) (parallel_generatedRoot_283.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_4 ctx) (0) (parallel_generatedRoot_283.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_4 ctx) (1) (parallel_generatedRoot_283.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_5 ctx) (0) (parallel_generatedRoot_283.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_283.w_6_5 ctx) (1) (parallel_generatedRoot_283.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_283_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_283.w_6_0 ctx), ((parallel_generatedRoot_283.w_6_1 ctx), ((parallel_generatedRoot_283.w_6_2 ctx), ((parallel_generatedRoot_283.w_6_3 ctx), ((parallel_generatedRoot_283.w_6_4 ctx), ((parallel_generatedRoot_283.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_283.w_9_0 ctx), ((parallel_generatedRoot_283.w_12_0 ctx), ((parallel_generatedRoot_283.w_15_0 ctx), ((parallel_generatedRoot_283.w_18_0 ctx), ((parallel_generatedRoot_283.w_21_0 ctx), ((parallel_generatedRoot_283.w_24_0 ctx), ())))))) ((parallel_generatedRoot_283.w_25_0 ctx), ((parallel_generatedRoot_283.w_25_1 ctx), ((parallel_generatedRoot_283.w_25_2 ctx), ((parallel_generatedRoot_283.w_25_3 ctx), ((parallel_generatedRoot_283.w_25_4 ctx), ((parallel_generatedRoot_283.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_283.outs ctx

theorem parallel_generatedRoot_283.facts_0  {ctx : parallel_generatedRoot_283Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_283.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_283.w_0_0 ctx) (parallel_generatedRoot_283.w_1_0 ctx) (parallel_generatedRoot_283.w_2_0 ctx) (parallel_generatedRoot_283.w_3_0 ctx) (parallel_generatedRoot_283.w_4_0 ctx) (parallel_generatedRoot_283.w_5_0 ctx) (parallel_generatedRoot_283.w_6_0 ctx) (parallel_generatedRoot_283.w_6_1 ctx) (parallel_generatedRoot_283.w_6_2 ctx) (parallel_generatedRoot_283.w_6_3 ctx) (parallel_generatedRoot_283.w_6_4 ctx) (parallel_generatedRoot_283.w_6_5 ctx) (parallel_generatedRoot_283.w_7_0 ctx) (parallel_generatedRoot_283.w_8_0 ctx) (parallel_generatedRoot_283.w_9_0 ctx) (parallel_generatedRoot_283.w_10_0 ctx) (parallel_generatedRoot_283.w_11_0 ctx) (parallel_generatedRoot_283.w_12_0 ctx) (parallel_generatedRoot_283.w_13_0 ctx) (parallel_generatedRoot_283.w_14_0 ctx) (parallel_generatedRoot_283.w_15_0 ctx) (parallel_generatedRoot_283.w_16_0 ctx) (parallel_generatedRoot_283.w_17_0 ctx) (parallel_generatedRoot_283.w_18_0 ctx) (parallel_generatedRoot_283.w_19_0 ctx) (parallel_generatedRoot_283.w_20_0 ctx) (parallel_generatedRoot_283.w_21_0 ctx) (parallel_generatedRoot_283.w_22_0 ctx) (parallel_generatedRoot_283.w_23_0 ctx) (parallel_generatedRoot_283.w_24_0 ctx) (parallel_generatedRoot_283.w_25_0 ctx) (parallel_generatedRoot_283.w_25_1 ctx) (parallel_generatedRoot_283.w_25_2 ctx) (parallel_generatedRoot_283.w_25_3 ctx) (parallel_generatedRoot_283.w_25_4 ctx) (parallel_generatedRoot_283.w_25_5 ctx) outputs) :
    parallel_generatedRoot_283.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_283.Facts  (ctx : parallel_generatedRoot_283Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_283.Facts_0 ctx outputs

theorem parallel_generatedRoot_283.facts  (ctx : parallel_generatedRoot_283Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_283.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_283.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_283.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_287.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_287.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_287Ctx" => (parallel_generatedRoot_287.Ctx backend hashModel params i_0)

def parallel_generatedRoot_287.w_0_0  (ctx : parallel_generatedRoot_287Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_287.w_1_0  (ctx : parallel_generatedRoot_287Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_287.w_2_0  (ctx : parallel_generatedRoot_287Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_287.w_3_0  (ctx : parallel_generatedRoot_287Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_287.w_4_0  (ctx : parallel_generatedRoot_287Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_287.w_5_0  (ctx : parallel_generatedRoot_287Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_287.w_6_0  (ctx : parallel_generatedRoot_287Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_287.w_6_1  (ctx : parallel_generatedRoot_287Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_287.w_6_2  (ctx : parallel_generatedRoot_287Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_287.w_6_3  (ctx : parallel_generatedRoot_287Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_287.w_6_4  (ctx : parallel_generatedRoot_287Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_287.w_6_5  (ctx : parallel_generatedRoot_287Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_287.w_7_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_287.w_8_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_287.w_9_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_287.w_7_0 ctx) (parallel_generatedRoot_287.w_8_0 ctx)

def parallel_generatedRoot_287.w_10_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_287.w_11_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_287.w_12_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_287.w_10_0 ctx) (parallel_generatedRoot_287.w_11_0 ctx)

def parallel_generatedRoot_287.w_13_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_287.w_14_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_287.w_15_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_287.w_13_0 ctx) (parallel_generatedRoot_287.w_14_0 ctx)

def parallel_generatedRoot_287.w_16_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_287.w_17_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_287.w_18_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_287.w_16_0 ctx) (parallel_generatedRoot_287.w_17_0 ctx)

def parallel_generatedRoot_287.w_19_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_287.w_20_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_287.w_21_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_287.w_19_0 ctx) (parallel_generatedRoot_287.w_20_0 ctx)

def parallel_generatedRoot_287.w_22_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_287.w_23_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_287.w_24_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_287.w_22_0 ctx) (parallel_generatedRoot_287.w_23_0 ctx)

def parallel_generatedRoot_287.w_25_0  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_287.w_25_1  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_287.w_25_2  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_287.w_25_3  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_287.w_25_4  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_287.w_25_5  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_287.outs  (ctx : parallel_generatedRoot_287Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_287.w_25_0 ctx), ((parallel_generatedRoot_287.w_25_1 ctx), ((parallel_generatedRoot_287.w_25_2 ctx), ((parallel_generatedRoot_287.w_25_3 ctx), ((parallel_generatedRoot_287.w_25_4 ctx), ((parallel_generatedRoot_287.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_287.Facts_0  (ctx : parallel_generatedRoot_287Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_287_6 backend hashModel params i_0 i ((parallel_generatedRoot_287.w_0_0 ctx), ((parallel_generatedRoot_287.w_1_0 ctx), ((parallel_generatedRoot_287.w_2_0 ctx), ((parallel_generatedRoot_287.w_3_0 ctx), ((parallel_generatedRoot_287.w_4_0 ctx), ((parallel_generatedRoot_287.w_5_0 ctx), ())))))) (((parallel_generatedRoot_287.w_6_0 ctx) i), (((parallel_generatedRoot_287.w_6_1 ctx) i), (((parallel_generatedRoot_287.w_6_2 ctx) i), (((parallel_generatedRoot_287.w_6_3 ctx) i), (((parallel_generatedRoot_287.w_6_4 ctx) i), (((parallel_generatedRoot_287.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_0 ctx) (0) (parallel_generatedRoot_287.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_0 ctx) (1) (parallel_generatedRoot_287.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_1 ctx) (0) (parallel_generatedRoot_287.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_1 ctx) (1) (parallel_generatedRoot_287.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_2 ctx) (0) (parallel_generatedRoot_287.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_2 ctx) (1) (parallel_generatedRoot_287.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_3 ctx) (0) (parallel_generatedRoot_287.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_3 ctx) (1) (parallel_generatedRoot_287.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_4 ctx) (0) (parallel_generatedRoot_287.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_4 ctx) (1) (parallel_generatedRoot_287.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_5 ctx) (0) (parallel_generatedRoot_287.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_287.w_6_5 ctx) (1) (parallel_generatedRoot_287.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_287_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_287.w_6_0 ctx), ((parallel_generatedRoot_287.w_6_1 ctx), ((parallel_generatedRoot_287.w_6_2 ctx), ((parallel_generatedRoot_287.w_6_3 ctx), ((parallel_generatedRoot_287.w_6_4 ctx), ((parallel_generatedRoot_287.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_287.w_9_0 ctx), ((parallel_generatedRoot_287.w_12_0 ctx), ((parallel_generatedRoot_287.w_15_0 ctx), ((parallel_generatedRoot_287.w_18_0 ctx), ((parallel_generatedRoot_287.w_21_0 ctx), ((parallel_generatedRoot_287.w_24_0 ctx), ())))))) ((parallel_generatedRoot_287.w_25_0 ctx), ((parallel_generatedRoot_287.w_25_1 ctx), ((parallel_generatedRoot_287.w_25_2 ctx), ((parallel_generatedRoot_287.w_25_3 ctx), ((parallel_generatedRoot_287.w_25_4 ctx), ((parallel_generatedRoot_287.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_287.outs ctx

theorem parallel_generatedRoot_287.facts_0  {ctx : parallel_generatedRoot_287Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_287.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_287.w_0_0 ctx) (parallel_generatedRoot_287.w_1_0 ctx) (parallel_generatedRoot_287.w_2_0 ctx) (parallel_generatedRoot_287.w_3_0 ctx) (parallel_generatedRoot_287.w_4_0 ctx) (parallel_generatedRoot_287.w_5_0 ctx) (parallel_generatedRoot_287.w_6_0 ctx) (parallel_generatedRoot_287.w_6_1 ctx) (parallel_generatedRoot_287.w_6_2 ctx) (parallel_generatedRoot_287.w_6_3 ctx) (parallel_generatedRoot_287.w_6_4 ctx) (parallel_generatedRoot_287.w_6_5 ctx) (parallel_generatedRoot_287.w_7_0 ctx) (parallel_generatedRoot_287.w_8_0 ctx) (parallel_generatedRoot_287.w_9_0 ctx) (parallel_generatedRoot_287.w_10_0 ctx) (parallel_generatedRoot_287.w_11_0 ctx) (parallel_generatedRoot_287.w_12_0 ctx) (parallel_generatedRoot_287.w_13_0 ctx) (parallel_generatedRoot_287.w_14_0 ctx) (parallel_generatedRoot_287.w_15_0 ctx) (parallel_generatedRoot_287.w_16_0 ctx) (parallel_generatedRoot_287.w_17_0 ctx) (parallel_generatedRoot_287.w_18_0 ctx) (parallel_generatedRoot_287.w_19_0 ctx) (parallel_generatedRoot_287.w_20_0 ctx) (parallel_generatedRoot_287.w_21_0 ctx) (parallel_generatedRoot_287.w_22_0 ctx) (parallel_generatedRoot_287.w_23_0 ctx) (parallel_generatedRoot_287.w_24_0 ctx) (parallel_generatedRoot_287.w_25_0 ctx) (parallel_generatedRoot_287.w_25_1 ctx) (parallel_generatedRoot_287.w_25_2 ctx) (parallel_generatedRoot_287.w_25_3 ctx) (parallel_generatedRoot_287.w_25_4 ctx) (parallel_generatedRoot_287.w_25_5 ctx) outputs) :
    parallel_generatedRoot_287.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_287.Facts  (ctx : parallel_generatedRoot_287Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_287.Facts_0 ctx outputs

theorem parallel_generatedRoot_287.facts  (ctx : parallel_generatedRoot_287Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_287.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_287.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_287.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_291.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_291.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_291Ctx" => (parallel_generatedRoot_291.Ctx backend hashModel params i_0)

def parallel_generatedRoot_291.w_0_0  (ctx : parallel_generatedRoot_291Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_291.w_1_0  (ctx : parallel_generatedRoot_291Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_291.w_2_0  (ctx : parallel_generatedRoot_291Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_291.w_3_0  (ctx : parallel_generatedRoot_291Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_291.w_4_0  (ctx : parallel_generatedRoot_291Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_291.w_5_0  (ctx : parallel_generatedRoot_291Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_291.w_6_0  (ctx : parallel_generatedRoot_291Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_291.w_6_1  (ctx : parallel_generatedRoot_291Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_291.w_6_2  (ctx : parallel_generatedRoot_291Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_291.w_6_3  (ctx : parallel_generatedRoot_291Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_291.w_6_4  (ctx : parallel_generatedRoot_291Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_291.w_6_5  (ctx : parallel_generatedRoot_291Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_291.w_7_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_291.w_8_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_291.w_9_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_291.w_7_0 ctx) (parallel_generatedRoot_291.w_8_0 ctx)

def parallel_generatedRoot_291.w_10_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_291.w_11_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_291.w_12_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_291.w_10_0 ctx) (parallel_generatedRoot_291.w_11_0 ctx)

def parallel_generatedRoot_291.w_13_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_291.w_14_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_291.w_15_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_291.w_13_0 ctx) (parallel_generatedRoot_291.w_14_0 ctx)

def parallel_generatedRoot_291.w_16_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_291.w_17_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_291.w_18_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_291.w_16_0 ctx) (parallel_generatedRoot_291.w_17_0 ctx)

def parallel_generatedRoot_291.w_19_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_291.w_20_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_291.w_21_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_291.w_19_0 ctx) (parallel_generatedRoot_291.w_20_0 ctx)

def parallel_generatedRoot_291.w_22_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_291.w_23_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_291.w_24_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_291.w_22_0 ctx) (parallel_generatedRoot_291.w_23_0 ctx)

def parallel_generatedRoot_291.w_25_0  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_291.w_25_1  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_291.w_25_2  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_291.w_25_3  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_291.w_25_4  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_291.w_25_5  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_291.outs  (ctx : parallel_generatedRoot_291Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_291.w_25_0 ctx), ((parallel_generatedRoot_291.w_25_1 ctx), ((parallel_generatedRoot_291.w_25_2 ctx), ((parallel_generatedRoot_291.w_25_3 ctx), ((parallel_generatedRoot_291.w_25_4 ctx), ((parallel_generatedRoot_291.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_291.Facts_0  (ctx : parallel_generatedRoot_291Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_291_6 backend hashModel params i_0 i ((parallel_generatedRoot_291.w_0_0 ctx), ((parallel_generatedRoot_291.w_1_0 ctx), ((parallel_generatedRoot_291.w_2_0 ctx), ((parallel_generatedRoot_291.w_3_0 ctx), ((parallel_generatedRoot_291.w_4_0 ctx), ((parallel_generatedRoot_291.w_5_0 ctx), ())))))) (((parallel_generatedRoot_291.w_6_0 ctx) i), (((parallel_generatedRoot_291.w_6_1 ctx) i), (((parallel_generatedRoot_291.w_6_2 ctx) i), (((parallel_generatedRoot_291.w_6_3 ctx) i), (((parallel_generatedRoot_291.w_6_4 ctx) i), (((parallel_generatedRoot_291.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_0 ctx) (0) (parallel_generatedRoot_291.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_0 ctx) (1) (parallel_generatedRoot_291.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_1 ctx) (0) (parallel_generatedRoot_291.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_1 ctx) (1) (parallel_generatedRoot_291.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_2 ctx) (0) (parallel_generatedRoot_291.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_2 ctx) (1) (parallel_generatedRoot_291.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_3 ctx) (0) (parallel_generatedRoot_291.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_3 ctx) (1) (parallel_generatedRoot_291.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_4 ctx) (0) (parallel_generatedRoot_291.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_4 ctx) (1) (parallel_generatedRoot_291.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_5 ctx) (0) (parallel_generatedRoot_291.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_291.w_6_5 ctx) (1) (parallel_generatedRoot_291.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_291_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_291.w_6_0 ctx), ((parallel_generatedRoot_291.w_6_1 ctx), ((parallel_generatedRoot_291.w_6_2 ctx), ((parallel_generatedRoot_291.w_6_3 ctx), ((parallel_generatedRoot_291.w_6_4 ctx), ((parallel_generatedRoot_291.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_291.w_9_0 ctx), ((parallel_generatedRoot_291.w_12_0 ctx), ((parallel_generatedRoot_291.w_15_0 ctx), ((parallel_generatedRoot_291.w_18_0 ctx), ((parallel_generatedRoot_291.w_21_0 ctx), ((parallel_generatedRoot_291.w_24_0 ctx), ())))))) ((parallel_generatedRoot_291.w_25_0 ctx), ((parallel_generatedRoot_291.w_25_1 ctx), ((parallel_generatedRoot_291.w_25_2 ctx), ((parallel_generatedRoot_291.w_25_3 ctx), ((parallel_generatedRoot_291.w_25_4 ctx), ((parallel_generatedRoot_291.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_291.outs ctx

theorem parallel_generatedRoot_291.facts_0  {ctx : parallel_generatedRoot_291Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_291.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_291.w_0_0 ctx) (parallel_generatedRoot_291.w_1_0 ctx) (parallel_generatedRoot_291.w_2_0 ctx) (parallel_generatedRoot_291.w_3_0 ctx) (parallel_generatedRoot_291.w_4_0 ctx) (parallel_generatedRoot_291.w_5_0 ctx) (parallel_generatedRoot_291.w_6_0 ctx) (parallel_generatedRoot_291.w_6_1 ctx) (parallel_generatedRoot_291.w_6_2 ctx) (parallel_generatedRoot_291.w_6_3 ctx) (parallel_generatedRoot_291.w_6_4 ctx) (parallel_generatedRoot_291.w_6_5 ctx) (parallel_generatedRoot_291.w_7_0 ctx) (parallel_generatedRoot_291.w_8_0 ctx) (parallel_generatedRoot_291.w_9_0 ctx) (parallel_generatedRoot_291.w_10_0 ctx) (parallel_generatedRoot_291.w_11_0 ctx) (parallel_generatedRoot_291.w_12_0 ctx) (parallel_generatedRoot_291.w_13_0 ctx) (parallel_generatedRoot_291.w_14_0 ctx) (parallel_generatedRoot_291.w_15_0 ctx) (parallel_generatedRoot_291.w_16_0 ctx) (parallel_generatedRoot_291.w_17_0 ctx) (parallel_generatedRoot_291.w_18_0 ctx) (parallel_generatedRoot_291.w_19_0 ctx) (parallel_generatedRoot_291.w_20_0 ctx) (parallel_generatedRoot_291.w_21_0 ctx) (parallel_generatedRoot_291.w_22_0 ctx) (parallel_generatedRoot_291.w_23_0 ctx) (parallel_generatedRoot_291.w_24_0 ctx) (parallel_generatedRoot_291.w_25_0 ctx) (parallel_generatedRoot_291.w_25_1 ctx) (parallel_generatedRoot_291.w_25_2 ctx) (parallel_generatedRoot_291.w_25_3 ctx) (parallel_generatedRoot_291.w_25_4 ctx) (parallel_generatedRoot_291.w_25_5 ctx) outputs) :
    parallel_generatedRoot_291.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_291.Facts  (ctx : parallel_generatedRoot_291Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_291.Facts_0 ctx outputs

theorem parallel_generatedRoot_291.facts  (ctx : parallel_generatedRoot_291Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_291.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_291.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_291.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_295.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_295.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_295Ctx" => (parallel_generatedRoot_295.Ctx backend hashModel params i_0)

def parallel_generatedRoot_295.w_0_0  (ctx : parallel_generatedRoot_295Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_295.w_1_0  (ctx : parallel_generatedRoot_295Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_295.w_2_0  (ctx : parallel_generatedRoot_295Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_295.w_3_0  (ctx : parallel_generatedRoot_295Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_295.w_4_0  (ctx : parallel_generatedRoot_295Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_295.w_5_0  (ctx : parallel_generatedRoot_295Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_295.w_6_0  (ctx : parallel_generatedRoot_295Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_295.w_6_1  (ctx : parallel_generatedRoot_295Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_295.w_6_2  (ctx : parallel_generatedRoot_295Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_295.w_6_3  (ctx : parallel_generatedRoot_295Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_295.w_6_4  (ctx : parallel_generatedRoot_295Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_295.w_6_5  (ctx : parallel_generatedRoot_295Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_295.w_7_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_295.w_8_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_295.w_9_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_295.w_7_0 ctx) (parallel_generatedRoot_295.w_8_0 ctx)

def parallel_generatedRoot_295.w_10_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_295.w_11_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_295.w_12_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_295.w_10_0 ctx) (parallel_generatedRoot_295.w_11_0 ctx)

def parallel_generatedRoot_295.w_13_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_295.w_14_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_295.w_15_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_295.w_13_0 ctx) (parallel_generatedRoot_295.w_14_0 ctx)

def parallel_generatedRoot_295.w_16_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_295.w_17_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_295.w_18_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_295.w_16_0 ctx) (parallel_generatedRoot_295.w_17_0 ctx)

def parallel_generatedRoot_295.w_19_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_295.w_20_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_295.w_21_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_295.w_19_0 ctx) (parallel_generatedRoot_295.w_20_0 ctx)

def parallel_generatedRoot_295.w_22_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_295.w_23_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_295.w_24_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_295.w_22_0 ctx) (parallel_generatedRoot_295.w_23_0 ctx)

def parallel_generatedRoot_295.w_25_0  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_295.w_25_1  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_295.w_25_2  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_295.w_25_3  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_295.w_25_4  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_295.w_25_5  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_295.outs  (ctx : parallel_generatedRoot_295Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_295.w_25_0 ctx), ((parallel_generatedRoot_295.w_25_1 ctx), ((parallel_generatedRoot_295.w_25_2 ctx), ((parallel_generatedRoot_295.w_25_3 ctx), ((parallel_generatedRoot_295.w_25_4 ctx), ((parallel_generatedRoot_295.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_295.Facts_0  (ctx : parallel_generatedRoot_295Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_295_6 backend hashModel params i_0 i ((parallel_generatedRoot_295.w_0_0 ctx), ((parallel_generatedRoot_295.w_1_0 ctx), ((parallel_generatedRoot_295.w_2_0 ctx), ((parallel_generatedRoot_295.w_3_0 ctx), ((parallel_generatedRoot_295.w_4_0 ctx), ((parallel_generatedRoot_295.w_5_0 ctx), ())))))) (((parallel_generatedRoot_295.w_6_0 ctx) i), (((parallel_generatedRoot_295.w_6_1 ctx) i), (((parallel_generatedRoot_295.w_6_2 ctx) i), (((parallel_generatedRoot_295.w_6_3 ctx) i), (((parallel_generatedRoot_295.w_6_4 ctx) i), (((parallel_generatedRoot_295.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_0 ctx) (0) (parallel_generatedRoot_295.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_0 ctx) (1) (parallel_generatedRoot_295.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_1 ctx) (0) (parallel_generatedRoot_295.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_1 ctx) (1) (parallel_generatedRoot_295.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_2 ctx) (0) (parallel_generatedRoot_295.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_2 ctx) (1) (parallel_generatedRoot_295.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_3 ctx) (0) (parallel_generatedRoot_295.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_3 ctx) (1) (parallel_generatedRoot_295.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_4 ctx) (0) (parallel_generatedRoot_295.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_4 ctx) (1) (parallel_generatedRoot_295.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_5 ctx) (0) (parallel_generatedRoot_295.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_295.w_6_5 ctx) (1) (parallel_generatedRoot_295.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_295_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_295.w_6_0 ctx), ((parallel_generatedRoot_295.w_6_1 ctx), ((parallel_generatedRoot_295.w_6_2 ctx), ((parallel_generatedRoot_295.w_6_3 ctx), ((parallel_generatedRoot_295.w_6_4 ctx), ((parallel_generatedRoot_295.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_295.w_9_0 ctx), ((parallel_generatedRoot_295.w_12_0 ctx), ((parallel_generatedRoot_295.w_15_0 ctx), ((parallel_generatedRoot_295.w_18_0 ctx), ((parallel_generatedRoot_295.w_21_0 ctx), ((parallel_generatedRoot_295.w_24_0 ctx), ())))))) ((parallel_generatedRoot_295.w_25_0 ctx), ((parallel_generatedRoot_295.w_25_1 ctx), ((parallel_generatedRoot_295.w_25_2 ctx), ((parallel_generatedRoot_295.w_25_3 ctx), ((parallel_generatedRoot_295.w_25_4 ctx), ((parallel_generatedRoot_295.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_295.outs ctx

theorem parallel_generatedRoot_295.facts_0  {ctx : parallel_generatedRoot_295Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_295.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_295.w_0_0 ctx) (parallel_generatedRoot_295.w_1_0 ctx) (parallel_generatedRoot_295.w_2_0 ctx) (parallel_generatedRoot_295.w_3_0 ctx) (parallel_generatedRoot_295.w_4_0 ctx) (parallel_generatedRoot_295.w_5_0 ctx) (parallel_generatedRoot_295.w_6_0 ctx) (parallel_generatedRoot_295.w_6_1 ctx) (parallel_generatedRoot_295.w_6_2 ctx) (parallel_generatedRoot_295.w_6_3 ctx) (parallel_generatedRoot_295.w_6_4 ctx) (parallel_generatedRoot_295.w_6_5 ctx) (parallel_generatedRoot_295.w_7_0 ctx) (parallel_generatedRoot_295.w_8_0 ctx) (parallel_generatedRoot_295.w_9_0 ctx) (parallel_generatedRoot_295.w_10_0 ctx) (parallel_generatedRoot_295.w_11_0 ctx) (parallel_generatedRoot_295.w_12_0 ctx) (parallel_generatedRoot_295.w_13_0 ctx) (parallel_generatedRoot_295.w_14_0 ctx) (parallel_generatedRoot_295.w_15_0 ctx) (parallel_generatedRoot_295.w_16_0 ctx) (parallel_generatedRoot_295.w_17_0 ctx) (parallel_generatedRoot_295.w_18_0 ctx) (parallel_generatedRoot_295.w_19_0 ctx) (parallel_generatedRoot_295.w_20_0 ctx) (parallel_generatedRoot_295.w_21_0 ctx) (parallel_generatedRoot_295.w_22_0 ctx) (parallel_generatedRoot_295.w_23_0 ctx) (parallel_generatedRoot_295.w_24_0 ctx) (parallel_generatedRoot_295.w_25_0 ctx) (parallel_generatedRoot_295.w_25_1 ctx) (parallel_generatedRoot_295.w_25_2 ctx) (parallel_generatedRoot_295.w_25_3 ctx) (parallel_generatedRoot_295.w_25_4 ctx) (parallel_generatedRoot_295.w_25_5 ctx) outputs) :
    parallel_generatedRoot_295.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_295.Facts  (ctx : parallel_generatedRoot_295Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_295.Facts_0 ctx outputs

theorem parallel_generatedRoot_295.facts  (ctx : parallel_generatedRoot_295Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_295.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_295.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_295.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_299.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_299.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_299Ctx" => (parallel_generatedRoot_299.Ctx backend hashModel params i_0)

def parallel_generatedRoot_299.w_0_0  (ctx : parallel_generatedRoot_299Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_299.w_1_0  (ctx : parallel_generatedRoot_299Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_299.w_2_0  (ctx : parallel_generatedRoot_299Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_299.w_3_0  (ctx : parallel_generatedRoot_299Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_299.w_4_0  (ctx : parallel_generatedRoot_299Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_299.w_5_0  (ctx : parallel_generatedRoot_299Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_299.w_6_0  (ctx : parallel_generatedRoot_299Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_299.w_6_1  (ctx : parallel_generatedRoot_299Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_299.w_6_2  (ctx : parallel_generatedRoot_299Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_299.w_6_3  (ctx : parallel_generatedRoot_299Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_299.w_6_4  (ctx : parallel_generatedRoot_299Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_299.w_6_5  (ctx : parallel_generatedRoot_299Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_299.w_7_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_299.w_8_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_299.w_9_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_299.w_7_0 ctx) (parallel_generatedRoot_299.w_8_0 ctx)

def parallel_generatedRoot_299.w_10_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_299.w_11_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_299.w_12_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_299.w_10_0 ctx) (parallel_generatedRoot_299.w_11_0 ctx)

def parallel_generatedRoot_299.w_13_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_299.w_14_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_299.w_15_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_299.w_13_0 ctx) (parallel_generatedRoot_299.w_14_0 ctx)

def parallel_generatedRoot_299.w_16_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_299.w_17_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_299.w_18_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_299.w_16_0 ctx) (parallel_generatedRoot_299.w_17_0 ctx)

def parallel_generatedRoot_299.w_19_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_299.w_20_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_299.w_21_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_299.w_19_0 ctx) (parallel_generatedRoot_299.w_20_0 ctx)

def parallel_generatedRoot_299.w_22_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_299.w_23_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_299.w_24_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_299.w_22_0 ctx) (parallel_generatedRoot_299.w_23_0 ctx)

def parallel_generatedRoot_299.w_25_0  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_299.w_25_1  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_299.w_25_2  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_299.w_25_3  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_299.w_25_4  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_299.w_25_5  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_299.outs  (ctx : parallel_generatedRoot_299Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_299.w_25_0 ctx), ((parallel_generatedRoot_299.w_25_1 ctx), ((parallel_generatedRoot_299.w_25_2 ctx), ((parallel_generatedRoot_299.w_25_3 ctx), ((parallel_generatedRoot_299.w_25_4 ctx), ((parallel_generatedRoot_299.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_299.Facts_0  (ctx : parallel_generatedRoot_299Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_299_6 backend hashModel params i_0 i ((parallel_generatedRoot_299.w_0_0 ctx), ((parallel_generatedRoot_299.w_1_0 ctx), ((parallel_generatedRoot_299.w_2_0 ctx), ((parallel_generatedRoot_299.w_3_0 ctx), ((parallel_generatedRoot_299.w_4_0 ctx), ((parallel_generatedRoot_299.w_5_0 ctx), ())))))) (((parallel_generatedRoot_299.w_6_0 ctx) i), (((parallel_generatedRoot_299.w_6_1 ctx) i), (((parallel_generatedRoot_299.w_6_2 ctx) i), (((parallel_generatedRoot_299.w_6_3 ctx) i), (((parallel_generatedRoot_299.w_6_4 ctx) i), (((parallel_generatedRoot_299.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_0 ctx) (0) (parallel_generatedRoot_299.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_0 ctx) (1) (parallel_generatedRoot_299.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_1 ctx) (0) (parallel_generatedRoot_299.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_1 ctx) (1) (parallel_generatedRoot_299.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_2 ctx) (0) (parallel_generatedRoot_299.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_2 ctx) (1) (parallel_generatedRoot_299.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_3 ctx) (0) (parallel_generatedRoot_299.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_3 ctx) (1) (parallel_generatedRoot_299.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_4 ctx) (0) (parallel_generatedRoot_299.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_4 ctx) (1) (parallel_generatedRoot_299.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_5 ctx) (0) (parallel_generatedRoot_299.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_299.w_6_5 ctx) (1) (parallel_generatedRoot_299.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_299_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_299.w_6_0 ctx), ((parallel_generatedRoot_299.w_6_1 ctx), ((parallel_generatedRoot_299.w_6_2 ctx), ((parallel_generatedRoot_299.w_6_3 ctx), ((parallel_generatedRoot_299.w_6_4 ctx), ((parallel_generatedRoot_299.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_299.w_9_0 ctx), ((parallel_generatedRoot_299.w_12_0 ctx), ((parallel_generatedRoot_299.w_15_0 ctx), ((parallel_generatedRoot_299.w_18_0 ctx), ((parallel_generatedRoot_299.w_21_0 ctx), ((parallel_generatedRoot_299.w_24_0 ctx), ())))))) ((parallel_generatedRoot_299.w_25_0 ctx), ((parallel_generatedRoot_299.w_25_1 ctx), ((parallel_generatedRoot_299.w_25_2 ctx), ((parallel_generatedRoot_299.w_25_3 ctx), ((parallel_generatedRoot_299.w_25_4 ctx), ((parallel_generatedRoot_299.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_299.outs ctx

theorem parallel_generatedRoot_299.facts_0  {ctx : parallel_generatedRoot_299Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_299.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_299.w_0_0 ctx) (parallel_generatedRoot_299.w_1_0 ctx) (parallel_generatedRoot_299.w_2_0 ctx) (parallel_generatedRoot_299.w_3_0 ctx) (parallel_generatedRoot_299.w_4_0 ctx) (parallel_generatedRoot_299.w_5_0 ctx) (parallel_generatedRoot_299.w_6_0 ctx) (parallel_generatedRoot_299.w_6_1 ctx) (parallel_generatedRoot_299.w_6_2 ctx) (parallel_generatedRoot_299.w_6_3 ctx) (parallel_generatedRoot_299.w_6_4 ctx) (parallel_generatedRoot_299.w_6_5 ctx) (parallel_generatedRoot_299.w_7_0 ctx) (parallel_generatedRoot_299.w_8_0 ctx) (parallel_generatedRoot_299.w_9_0 ctx) (parallel_generatedRoot_299.w_10_0 ctx) (parallel_generatedRoot_299.w_11_0 ctx) (parallel_generatedRoot_299.w_12_0 ctx) (parallel_generatedRoot_299.w_13_0 ctx) (parallel_generatedRoot_299.w_14_0 ctx) (parallel_generatedRoot_299.w_15_0 ctx) (parallel_generatedRoot_299.w_16_0 ctx) (parallel_generatedRoot_299.w_17_0 ctx) (parallel_generatedRoot_299.w_18_0 ctx) (parallel_generatedRoot_299.w_19_0 ctx) (parallel_generatedRoot_299.w_20_0 ctx) (parallel_generatedRoot_299.w_21_0 ctx) (parallel_generatedRoot_299.w_22_0 ctx) (parallel_generatedRoot_299.w_23_0 ctx) (parallel_generatedRoot_299.w_24_0 ctx) (parallel_generatedRoot_299.w_25_0 ctx) (parallel_generatedRoot_299.w_25_1 ctx) (parallel_generatedRoot_299.w_25_2 ctx) (parallel_generatedRoot_299.w_25_3 ctx) (parallel_generatedRoot_299.w_25_4 ctx) (parallel_generatedRoot_299.w_25_5 ctx) outputs) :
    parallel_generatedRoot_299.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_299.Facts  (ctx : parallel_generatedRoot_299Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_299.Facts_0 ctx outputs

theorem parallel_generatedRoot_299.facts  (ctx : parallel_generatedRoot_299Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_299.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_299.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_299.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_303.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_303.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_303Ctx" => (parallel_generatedRoot_303.Ctx backend hashModel params i_0)

def parallel_generatedRoot_303.w_0_0  (ctx : parallel_generatedRoot_303Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_303.w_1_0  (ctx : parallel_generatedRoot_303Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_303.w_2_0  (ctx : parallel_generatedRoot_303Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_303.w_3_0  (ctx : parallel_generatedRoot_303Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_303.w_4_0  (ctx : parallel_generatedRoot_303Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_303.w_5_0  (ctx : parallel_generatedRoot_303Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_303.w_6_0  (ctx : parallel_generatedRoot_303Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_303.w_6_1  (ctx : parallel_generatedRoot_303Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_303.w_6_2  (ctx : parallel_generatedRoot_303Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_303.w_6_3  (ctx : parallel_generatedRoot_303Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_303.w_6_4  (ctx : parallel_generatedRoot_303Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_303.w_6_5  (ctx : parallel_generatedRoot_303Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_303.w_7_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_303.w_8_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_303.w_9_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_303.w_7_0 ctx) (parallel_generatedRoot_303.w_8_0 ctx)

def parallel_generatedRoot_303.w_10_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_303.w_11_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_303.w_12_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_303.w_10_0 ctx) (parallel_generatedRoot_303.w_11_0 ctx)

def parallel_generatedRoot_303.w_13_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_303.w_14_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_303.w_15_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_303.w_13_0 ctx) (parallel_generatedRoot_303.w_14_0 ctx)

def parallel_generatedRoot_303.w_16_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_303.w_17_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_303.w_18_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_303.w_16_0 ctx) (parallel_generatedRoot_303.w_17_0 ctx)

def parallel_generatedRoot_303.w_19_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_303.w_20_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_303.w_21_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_303.w_19_0 ctx) (parallel_generatedRoot_303.w_20_0 ctx)

def parallel_generatedRoot_303.w_22_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_303.w_23_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_303.w_24_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_303.w_22_0 ctx) (parallel_generatedRoot_303.w_23_0 ctx)

def parallel_generatedRoot_303.w_25_0  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_303.w_25_1  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_303.w_25_2  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_303.w_25_3  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_303.w_25_4  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_303.w_25_5  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_303.outs  (ctx : parallel_generatedRoot_303Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_303.w_25_0 ctx), ((parallel_generatedRoot_303.w_25_1 ctx), ((parallel_generatedRoot_303.w_25_2 ctx), ((parallel_generatedRoot_303.w_25_3 ctx), ((parallel_generatedRoot_303.w_25_4 ctx), ((parallel_generatedRoot_303.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_303.Facts_0  (ctx : parallel_generatedRoot_303Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_303_6 backend hashModel params i_0 i ((parallel_generatedRoot_303.w_0_0 ctx), ((parallel_generatedRoot_303.w_1_0 ctx), ((parallel_generatedRoot_303.w_2_0 ctx), ((parallel_generatedRoot_303.w_3_0 ctx), ((parallel_generatedRoot_303.w_4_0 ctx), ((parallel_generatedRoot_303.w_5_0 ctx), ())))))) (((parallel_generatedRoot_303.w_6_0 ctx) i), (((parallel_generatedRoot_303.w_6_1 ctx) i), (((parallel_generatedRoot_303.w_6_2 ctx) i), (((parallel_generatedRoot_303.w_6_3 ctx) i), (((parallel_generatedRoot_303.w_6_4 ctx) i), (((parallel_generatedRoot_303.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_0 ctx) (0) (parallel_generatedRoot_303.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_0 ctx) (1) (parallel_generatedRoot_303.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_1 ctx) (0) (parallel_generatedRoot_303.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_1 ctx) (1) (parallel_generatedRoot_303.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_2 ctx) (0) (parallel_generatedRoot_303.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_2 ctx) (1) (parallel_generatedRoot_303.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_3 ctx) (0) (parallel_generatedRoot_303.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_3 ctx) (1) (parallel_generatedRoot_303.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_4 ctx) (0) (parallel_generatedRoot_303.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_4 ctx) (1) (parallel_generatedRoot_303.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_5 ctx) (0) (parallel_generatedRoot_303.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_303.w_6_5 ctx) (1) (parallel_generatedRoot_303.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_303_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_303.w_6_0 ctx), ((parallel_generatedRoot_303.w_6_1 ctx), ((parallel_generatedRoot_303.w_6_2 ctx), ((parallel_generatedRoot_303.w_6_3 ctx), ((parallel_generatedRoot_303.w_6_4 ctx), ((parallel_generatedRoot_303.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_303.w_9_0 ctx), ((parallel_generatedRoot_303.w_12_0 ctx), ((parallel_generatedRoot_303.w_15_0 ctx), ((parallel_generatedRoot_303.w_18_0 ctx), ((parallel_generatedRoot_303.w_21_0 ctx), ((parallel_generatedRoot_303.w_24_0 ctx), ())))))) ((parallel_generatedRoot_303.w_25_0 ctx), ((parallel_generatedRoot_303.w_25_1 ctx), ((parallel_generatedRoot_303.w_25_2 ctx), ((parallel_generatedRoot_303.w_25_3 ctx), ((parallel_generatedRoot_303.w_25_4 ctx), ((parallel_generatedRoot_303.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_303.outs ctx

theorem parallel_generatedRoot_303.facts_0  {ctx : parallel_generatedRoot_303Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_303.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_303.w_0_0 ctx) (parallel_generatedRoot_303.w_1_0 ctx) (parallel_generatedRoot_303.w_2_0 ctx) (parallel_generatedRoot_303.w_3_0 ctx) (parallel_generatedRoot_303.w_4_0 ctx) (parallel_generatedRoot_303.w_5_0 ctx) (parallel_generatedRoot_303.w_6_0 ctx) (parallel_generatedRoot_303.w_6_1 ctx) (parallel_generatedRoot_303.w_6_2 ctx) (parallel_generatedRoot_303.w_6_3 ctx) (parallel_generatedRoot_303.w_6_4 ctx) (parallel_generatedRoot_303.w_6_5 ctx) (parallel_generatedRoot_303.w_7_0 ctx) (parallel_generatedRoot_303.w_8_0 ctx) (parallel_generatedRoot_303.w_9_0 ctx) (parallel_generatedRoot_303.w_10_0 ctx) (parallel_generatedRoot_303.w_11_0 ctx) (parallel_generatedRoot_303.w_12_0 ctx) (parallel_generatedRoot_303.w_13_0 ctx) (parallel_generatedRoot_303.w_14_0 ctx) (parallel_generatedRoot_303.w_15_0 ctx) (parallel_generatedRoot_303.w_16_0 ctx) (parallel_generatedRoot_303.w_17_0 ctx) (parallel_generatedRoot_303.w_18_0 ctx) (parallel_generatedRoot_303.w_19_0 ctx) (parallel_generatedRoot_303.w_20_0 ctx) (parallel_generatedRoot_303.w_21_0 ctx) (parallel_generatedRoot_303.w_22_0 ctx) (parallel_generatedRoot_303.w_23_0 ctx) (parallel_generatedRoot_303.w_24_0 ctx) (parallel_generatedRoot_303.w_25_0 ctx) (parallel_generatedRoot_303.w_25_1 ctx) (parallel_generatedRoot_303.w_25_2 ctx) (parallel_generatedRoot_303.w_25_3 ctx) (parallel_generatedRoot_303.w_25_4 ctx) (parallel_generatedRoot_303.w_25_5 ctx) outputs) :
    parallel_generatedRoot_303.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_303.Facts  (ctx : parallel_generatedRoot_303Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_303.Facts_0 ctx outputs

theorem parallel_generatedRoot_303.facts  (ctx : parallel_generatedRoot_303Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_303.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_303.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_303.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_307.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_307.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_307Ctx" => (parallel_generatedRoot_307.Ctx backend hashModel params i_0)

def parallel_generatedRoot_307.w_0_0  (ctx : parallel_generatedRoot_307Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_307.w_1_0  (ctx : parallel_generatedRoot_307Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_307.w_2_0  (ctx : parallel_generatedRoot_307Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_307.w_3_0  (ctx : parallel_generatedRoot_307Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_307.w_4_0  (ctx : parallel_generatedRoot_307Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_307.w_5_0  (ctx : parallel_generatedRoot_307Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_307.w_6_0  (ctx : parallel_generatedRoot_307Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_307.w_6_1  (ctx : parallel_generatedRoot_307Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_307.w_6_2  (ctx : parallel_generatedRoot_307Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_307.w_6_3  (ctx : parallel_generatedRoot_307Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_307.w_6_4  (ctx : parallel_generatedRoot_307Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_307.w_6_5  (ctx : parallel_generatedRoot_307Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_307.w_7_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_307.w_8_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_307.w_9_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_307.w_7_0 ctx) (parallel_generatedRoot_307.w_8_0 ctx)

def parallel_generatedRoot_307.w_10_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_307.w_11_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_307.w_12_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_307.w_10_0 ctx) (parallel_generatedRoot_307.w_11_0 ctx)

def parallel_generatedRoot_307.w_13_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_307.w_14_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_307.w_15_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_307.w_13_0 ctx) (parallel_generatedRoot_307.w_14_0 ctx)

def parallel_generatedRoot_307.w_16_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_307.w_17_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_307.w_18_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_307.w_16_0 ctx) (parallel_generatedRoot_307.w_17_0 ctx)

def parallel_generatedRoot_307.w_19_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_307.w_20_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_307.w_21_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_307.w_19_0 ctx) (parallel_generatedRoot_307.w_20_0 ctx)

def parallel_generatedRoot_307.w_22_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_307.w_23_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_307.w_24_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_307.w_22_0 ctx) (parallel_generatedRoot_307.w_23_0 ctx)

def parallel_generatedRoot_307.w_25_0  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_307.w_25_1  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_307.w_25_2  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_307.w_25_3  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_307.w_25_4  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_307.w_25_5  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_307.outs  (ctx : parallel_generatedRoot_307Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_307.w_25_0 ctx), ((parallel_generatedRoot_307.w_25_1 ctx), ((parallel_generatedRoot_307.w_25_2 ctx), ((parallel_generatedRoot_307.w_25_3 ctx), ((parallel_generatedRoot_307.w_25_4 ctx), ((parallel_generatedRoot_307.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_307.Facts_0  (ctx : parallel_generatedRoot_307Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_307_6 backend hashModel params i_0 i ((parallel_generatedRoot_307.w_0_0 ctx), ((parallel_generatedRoot_307.w_1_0 ctx), ((parallel_generatedRoot_307.w_2_0 ctx), ((parallel_generatedRoot_307.w_3_0 ctx), ((parallel_generatedRoot_307.w_4_0 ctx), ((parallel_generatedRoot_307.w_5_0 ctx), ())))))) (((parallel_generatedRoot_307.w_6_0 ctx) i), (((parallel_generatedRoot_307.w_6_1 ctx) i), (((parallel_generatedRoot_307.w_6_2 ctx) i), (((parallel_generatedRoot_307.w_6_3 ctx) i), (((parallel_generatedRoot_307.w_6_4 ctx) i), (((parallel_generatedRoot_307.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_0 ctx) (0) (parallel_generatedRoot_307.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_0 ctx) (1) (parallel_generatedRoot_307.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_1 ctx) (0) (parallel_generatedRoot_307.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_1 ctx) (1) (parallel_generatedRoot_307.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_2 ctx) (0) (parallel_generatedRoot_307.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_2 ctx) (1) (parallel_generatedRoot_307.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_3 ctx) (0) (parallel_generatedRoot_307.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_3 ctx) (1) (parallel_generatedRoot_307.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_4 ctx) (0) (parallel_generatedRoot_307.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_4 ctx) (1) (parallel_generatedRoot_307.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_5 ctx) (0) (parallel_generatedRoot_307.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_307.w_6_5 ctx) (1) (parallel_generatedRoot_307.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_307_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_307.w_6_0 ctx), ((parallel_generatedRoot_307.w_6_1 ctx), ((parallel_generatedRoot_307.w_6_2 ctx), ((parallel_generatedRoot_307.w_6_3 ctx), ((parallel_generatedRoot_307.w_6_4 ctx), ((parallel_generatedRoot_307.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_307.w_9_0 ctx), ((parallel_generatedRoot_307.w_12_0 ctx), ((parallel_generatedRoot_307.w_15_0 ctx), ((parallel_generatedRoot_307.w_18_0 ctx), ((parallel_generatedRoot_307.w_21_0 ctx), ((parallel_generatedRoot_307.w_24_0 ctx), ())))))) ((parallel_generatedRoot_307.w_25_0 ctx), ((parallel_generatedRoot_307.w_25_1 ctx), ((parallel_generatedRoot_307.w_25_2 ctx), ((parallel_generatedRoot_307.w_25_3 ctx), ((parallel_generatedRoot_307.w_25_4 ctx), ((parallel_generatedRoot_307.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_307.outs ctx

theorem parallel_generatedRoot_307.facts_0  {ctx : parallel_generatedRoot_307Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_307.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_307.w_0_0 ctx) (parallel_generatedRoot_307.w_1_0 ctx) (parallel_generatedRoot_307.w_2_0 ctx) (parallel_generatedRoot_307.w_3_0 ctx) (parallel_generatedRoot_307.w_4_0 ctx) (parallel_generatedRoot_307.w_5_0 ctx) (parallel_generatedRoot_307.w_6_0 ctx) (parallel_generatedRoot_307.w_6_1 ctx) (parallel_generatedRoot_307.w_6_2 ctx) (parallel_generatedRoot_307.w_6_3 ctx) (parallel_generatedRoot_307.w_6_4 ctx) (parallel_generatedRoot_307.w_6_5 ctx) (parallel_generatedRoot_307.w_7_0 ctx) (parallel_generatedRoot_307.w_8_0 ctx) (parallel_generatedRoot_307.w_9_0 ctx) (parallel_generatedRoot_307.w_10_0 ctx) (parallel_generatedRoot_307.w_11_0 ctx) (parallel_generatedRoot_307.w_12_0 ctx) (parallel_generatedRoot_307.w_13_0 ctx) (parallel_generatedRoot_307.w_14_0 ctx) (parallel_generatedRoot_307.w_15_0 ctx) (parallel_generatedRoot_307.w_16_0 ctx) (parallel_generatedRoot_307.w_17_0 ctx) (parallel_generatedRoot_307.w_18_0 ctx) (parallel_generatedRoot_307.w_19_0 ctx) (parallel_generatedRoot_307.w_20_0 ctx) (parallel_generatedRoot_307.w_21_0 ctx) (parallel_generatedRoot_307.w_22_0 ctx) (parallel_generatedRoot_307.w_23_0 ctx) (parallel_generatedRoot_307.w_24_0 ctx) (parallel_generatedRoot_307.w_25_0 ctx) (parallel_generatedRoot_307.w_25_1 ctx) (parallel_generatedRoot_307.w_25_2 ctx) (parallel_generatedRoot_307.w_25_3 ctx) (parallel_generatedRoot_307.w_25_4 ctx) (parallel_generatedRoot_307.w_25_5 ctx) outputs) :
    parallel_generatedRoot_307.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_307.Facts  (ctx : parallel_generatedRoot_307Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_307.Facts_0 ctx outputs

theorem parallel_generatedRoot_307.facts  (ctx : parallel_generatedRoot_307Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_307.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_307.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_307.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_311.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_311.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_311Ctx" => (parallel_generatedRoot_311.Ctx backend hashModel params i_0)

def parallel_generatedRoot_311.w_0_0  (ctx : parallel_generatedRoot_311Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_311.w_1_0  (ctx : parallel_generatedRoot_311Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_311.w_2_0  (ctx : parallel_generatedRoot_311Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_311.w_3_0  (ctx : parallel_generatedRoot_311Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_311.w_4_0  (ctx : parallel_generatedRoot_311Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_311.w_5_0  (ctx : parallel_generatedRoot_311Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_311.w_6_0  (ctx : parallel_generatedRoot_311Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_311.w_6_1  (ctx : parallel_generatedRoot_311Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_311.w_6_2  (ctx : parallel_generatedRoot_311Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_311.w_6_3  (ctx : parallel_generatedRoot_311Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_311.w_6_4  (ctx : parallel_generatedRoot_311Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_311.w_6_5  (ctx : parallel_generatedRoot_311Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_311.w_7_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_311.w_8_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_311.w_9_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_311.w_7_0 ctx) (parallel_generatedRoot_311.w_8_0 ctx)

def parallel_generatedRoot_311.w_10_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_311.w_11_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_311.w_12_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_311.w_10_0 ctx) (parallel_generatedRoot_311.w_11_0 ctx)

def parallel_generatedRoot_311.w_13_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_311.w_14_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_311.w_15_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_311.w_13_0 ctx) (parallel_generatedRoot_311.w_14_0 ctx)

def parallel_generatedRoot_311.w_16_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_311.w_17_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_311.w_18_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_311.w_16_0 ctx) (parallel_generatedRoot_311.w_17_0 ctx)

def parallel_generatedRoot_311.w_19_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_311.w_20_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_311.w_21_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_311.w_19_0 ctx) (parallel_generatedRoot_311.w_20_0 ctx)

def parallel_generatedRoot_311.w_22_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_311.w_23_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_311.w_24_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_311.w_22_0 ctx) (parallel_generatedRoot_311.w_23_0 ctx)

def parallel_generatedRoot_311.w_25_0  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_311.w_25_1  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_311.w_25_2  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_311.w_25_3  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_311.w_25_4  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_311.w_25_5  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_311.outs  (ctx : parallel_generatedRoot_311Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_311.w_25_0 ctx), ((parallel_generatedRoot_311.w_25_1 ctx), ((parallel_generatedRoot_311.w_25_2 ctx), ((parallel_generatedRoot_311.w_25_3 ctx), ((parallel_generatedRoot_311.w_25_4 ctx), ((parallel_generatedRoot_311.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_311.Facts_0  (ctx : parallel_generatedRoot_311Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_311_6 backend hashModel params i_0 i ((parallel_generatedRoot_311.w_0_0 ctx), ((parallel_generatedRoot_311.w_1_0 ctx), ((parallel_generatedRoot_311.w_2_0 ctx), ((parallel_generatedRoot_311.w_3_0 ctx), ((parallel_generatedRoot_311.w_4_0 ctx), ((parallel_generatedRoot_311.w_5_0 ctx), ())))))) (((parallel_generatedRoot_311.w_6_0 ctx) i), (((parallel_generatedRoot_311.w_6_1 ctx) i), (((parallel_generatedRoot_311.w_6_2 ctx) i), (((parallel_generatedRoot_311.w_6_3 ctx) i), (((parallel_generatedRoot_311.w_6_4 ctx) i), (((parallel_generatedRoot_311.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_0 ctx) (0) (parallel_generatedRoot_311.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_0 ctx) (1) (parallel_generatedRoot_311.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_1 ctx) (0) (parallel_generatedRoot_311.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_1 ctx) (1) (parallel_generatedRoot_311.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_2 ctx) (0) (parallel_generatedRoot_311.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_2 ctx) (1) (parallel_generatedRoot_311.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_3 ctx) (0) (parallel_generatedRoot_311.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_3 ctx) (1) (parallel_generatedRoot_311.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_4 ctx) (0) (parallel_generatedRoot_311.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_4 ctx) (1) (parallel_generatedRoot_311.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_5 ctx) (0) (parallel_generatedRoot_311.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_311.w_6_5 ctx) (1) (parallel_generatedRoot_311.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_311_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_311.w_6_0 ctx), ((parallel_generatedRoot_311.w_6_1 ctx), ((parallel_generatedRoot_311.w_6_2 ctx), ((parallel_generatedRoot_311.w_6_3 ctx), ((parallel_generatedRoot_311.w_6_4 ctx), ((parallel_generatedRoot_311.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_311.w_9_0 ctx), ((parallel_generatedRoot_311.w_12_0 ctx), ((parallel_generatedRoot_311.w_15_0 ctx), ((parallel_generatedRoot_311.w_18_0 ctx), ((parallel_generatedRoot_311.w_21_0 ctx), ((parallel_generatedRoot_311.w_24_0 ctx), ())))))) ((parallel_generatedRoot_311.w_25_0 ctx), ((parallel_generatedRoot_311.w_25_1 ctx), ((parallel_generatedRoot_311.w_25_2 ctx), ((parallel_generatedRoot_311.w_25_3 ctx), ((parallel_generatedRoot_311.w_25_4 ctx), ((parallel_generatedRoot_311.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_311.outs ctx

theorem parallel_generatedRoot_311.facts_0  {ctx : parallel_generatedRoot_311Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_311.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_311.w_0_0 ctx) (parallel_generatedRoot_311.w_1_0 ctx) (parallel_generatedRoot_311.w_2_0 ctx) (parallel_generatedRoot_311.w_3_0 ctx) (parallel_generatedRoot_311.w_4_0 ctx) (parallel_generatedRoot_311.w_5_0 ctx) (parallel_generatedRoot_311.w_6_0 ctx) (parallel_generatedRoot_311.w_6_1 ctx) (parallel_generatedRoot_311.w_6_2 ctx) (parallel_generatedRoot_311.w_6_3 ctx) (parallel_generatedRoot_311.w_6_4 ctx) (parallel_generatedRoot_311.w_6_5 ctx) (parallel_generatedRoot_311.w_7_0 ctx) (parallel_generatedRoot_311.w_8_0 ctx) (parallel_generatedRoot_311.w_9_0 ctx) (parallel_generatedRoot_311.w_10_0 ctx) (parallel_generatedRoot_311.w_11_0 ctx) (parallel_generatedRoot_311.w_12_0 ctx) (parallel_generatedRoot_311.w_13_0 ctx) (parallel_generatedRoot_311.w_14_0 ctx) (parallel_generatedRoot_311.w_15_0 ctx) (parallel_generatedRoot_311.w_16_0 ctx) (parallel_generatedRoot_311.w_17_0 ctx) (parallel_generatedRoot_311.w_18_0 ctx) (parallel_generatedRoot_311.w_19_0 ctx) (parallel_generatedRoot_311.w_20_0 ctx) (parallel_generatedRoot_311.w_21_0 ctx) (parallel_generatedRoot_311.w_22_0 ctx) (parallel_generatedRoot_311.w_23_0 ctx) (parallel_generatedRoot_311.w_24_0 ctx) (parallel_generatedRoot_311.w_25_0 ctx) (parallel_generatedRoot_311.w_25_1 ctx) (parallel_generatedRoot_311.w_25_2 ctx) (parallel_generatedRoot_311.w_25_3 ctx) (parallel_generatedRoot_311.w_25_4 ctx) (parallel_generatedRoot_311.w_25_5 ctx) outputs) :
    parallel_generatedRoot_311.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_311.Facts  (ctx : parallel_generatedRoot_311Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_311.Facts_0 ctx outputs

theorem parallel_generatedRoot_311.facts  (ctx : parallel_generatedRoot_311Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_311.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_311.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_311.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_315.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_315.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_315Ctx" => (parallel_generatedRoot_315.Ctx backend hashModel params i_0)

def parallel_generatedRoot_315.w_0_0  (ctx : parallel_generatedRoot_315Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_315.w_1_0  (ctx : parallel_generatedRoot_315Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_315.w_2_0  (ctx : parallel_generatedRoot_315Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_315.w_3_0  (ctx : parallel_generatedRoot_315Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_315.w_4_0  (ctx : parallel_generatedRoot_315Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_315.w_5_0  (ctx : parallel_generatedRoot_315Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_315.w_6_0  (ctx : parallel_generatedRoot_315Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_315.w_6_1  (ctx : parallel_generatedRoot_315Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_315.w_6_2  (ctx : parallel_generatedRoot_315Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_315.w_6_3  (ctx : parallel_generatedRoot_315Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_315.w_6_4  (ctx : parallel_generatedRoot_315Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_315.w_6_5  (ctx : parallel_generatedRoot_315Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_315.w_7_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_315.w_8_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_315.w_9_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_315.w_7_0 ctx) (parallel_generatedRoot_315.w_8_0 ctx)

def parallel_generatedRoot_315.w_10_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_315.w_11_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_315.w_12_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_315.w_10_0 ctx) (parallel_generatedRoot_315.w_11_0 ctx)

def parallel_generatedRoot_315.w_13_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_315.w_14_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_315.w_15_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_315.w_13_0 ctx) (parallel_generatedRoot_315.w_14_0 ctx)

def parallel_generatedRoot_315.w_16_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_315.w_17_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_315.w_18_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_315.w_16_0 ctx) (parallel_generatedRoot_315.w_17_0 ctx)

def parallel_generatedRoot_315.w_19_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_315.w_20_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_315.w_21_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_315.w_19_0 ctx) (parallel_generatedRoot_315.w_20_0 ctx)

def parallel_generatedRoot_315.w_22_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_315.w_23_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_315.w_24_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_315.w_22_0 ctx) (parallel_generatedRoot_315.w_23_0 ctx)

def parallel_generatedRoot_315.w_25_0  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_315.w_25_1  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_315.w_25_2  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_315.w_25_3  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_315.w_25_4  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_315.w_25_5  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_315.outs  (ctx : parallel_generatedRoot_315Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_315.w_25_0 ctx), ((parallel_generatedRoot_315.w_25_1 ctx), ((parallel_generatedRoot_315.w_25_2 ctx), ((parallel_generatedRoot_315.w_25_3 ctx), ((parallel_generatedRoot_315.w_25_4 ctx), ((parallel_generatedRoot_315.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_315.Facts_0  (ctx : parallel_generatedRoot_315Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_315_6 backend hashModel params i_0 i ((parallel_generatedRoot_315.w_0_0 ctx), ((parallel_generatedRoot_315.w_1_0 ctx), ((parallel_generatedRoot_315.w_2_0 ctx), ((parallel_generatedRoot_315.w_3_0 ctx), ((parallel_generatedRoot_315.w_4_0 ctx), ((parallel_generatedRoot_315.w_5_0 ctx), ())))))) (((parallel_generatedRoot_315.w_6_0 ctx) i), (((parallel_generatedRoot_315.w_6_1 ctx) i), (((parallel_generatedRoot_315.w_6_2 ctx) i), (((parallel_generatedRoot_315.w_6_3 ctx) i), (((parallel_generatedRoot_315.w_6_4 ctx) i), (((parallel_generatedRoot_315.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_0 ctx) (0) (parallel_generatedRoot_315.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_0 ctx) (1) (parallel_generatedRoot_315.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_1 ctx) (0) (parallel_generatedRoot_315.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_1 ctx) (1) (parallel_generatedRoot_315.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_2 ctx) (0) (parallel_generatedRoot_315.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_2 ctx) (1) (parallel_generatedRoot_315.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_3 ctx) (0) (parallel_generatedRoot_315.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_3 ctx) (1) (parallel_generatedRoot_315.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_4 ctx) (0) (parallel_generatedRoot_315.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_4 ctx) (1) (parallel_generatedRoot_315.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_5 ctx) (0) (parallel_generatedRoot_315.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_315.w_6_5 ctx) (1) (parallel_generatedRoot_315.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_315_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_315.w_6_0 ctx), ((parallel_generatedRoot_315.w_6_1 ctx), ((parallel_generatedRoot_315.w_6_2 ctx), ((parallel_generatedRoot_315.w_6_3 ctx), ((parallel_generatedRoot_315.w_6_4 ctx), ((parallel_generatedRoot_315.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_315.w_9_0 ctx), ((parallel_generatedRoot_315.w_12_0 ctx), ((parallel_generatedRoot_315.w_15_0 ctx), ((parallel_generatedRoot_315.w_18_0 ctx), ((parallel_generatedRoot_315.w_21_0 ctx), ((parallel_generatedRoot_315.w_24_0 ctx), ())))))) ((parallel_generatedRoot_315.w_25_0 ctx), ((parallel_generatedRoot_315.w_25_1 ctx), ((parallel_generatedRoot_315.w_25_2 ctx), ((parallel_generatedRoot_315.w_25_3 ctx), ((parallel_generatedRoot_315.w_25_4 ctx), ((parallel_generatedRoot_315.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_315.outs ctx

theorem parallel_generatedRoot_315.facts_0  {ctx : parallel_generatedRoot_315Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_315.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_315.w_0_0 ctx) (parallel_generatedRoot_315.w_1_0 ctx) (parallel_generatedRoot_315.w_2_0 ctx) (parallel_generatedRoot_315.w_3_0 ctx) (parallel_generatedRoot_315.w_4_0 ctx) (parallel_generatedRoot_315.w_5_0 ctx) (parallel_generatedRoot_315.w_6_0 ctx) (parallel_generatedRoot_315.w_6_1 ctx) (parallel_generatedRoot_315.w_6_2 ctx) (parallel_generatedRoot_315.w_6_3 ctx) (parallel_generatedRoot_315.w_6_4 ctx) (parallel_generatedRoot_315.w_6_5 ctx) (parallel_generatedRoot_315.w_7_0 ctx) (parallel_generatedRoot_315.w_8_0 ctx) (parallel_generatedRoot_315.w_9_0 ctx) (parallel_generatedRoot_315.w_10_0 ctx) (parallel_generatedRoot_315.w_11_0 ctx) (parallel_generatedRoot_315.w_12_0 ctx) (parallel_generatedRoot_315.w_13_0 ctx) (parallel_generatedRoot_315.w_14_0 ctx) (parallel_generatedRoot_315.w_15_0 ctx) (parallel_generatedRoot_315.w_16_0 ctx) (parallel_generatedRoot_315.w_17_0 ctx) (parallel_generatedRoot_315.w_18_0 ctx) (parallel_generatedRoot_315.w_19_0 ctx) (parallel_generatedRoot_315.w_20_0 ctx) (parallel_generatedRoot_315.w_21_0 ctx) (parallel_generatedRoot_315.w_22_0 ctx) (parallel_generatedRoot_315.w_23_0 ctx) (parallel_generatedRoot_315.w_24_0 ctx) (parallel_generatedRoot_315.w_25_0 ctx) (parallel_generatedRoot_315.w_25_1 ctx) (parallel_generatedRoot_315.w_25_2 ctx) (parallel_generatedRoot_315.w_25_3 ctx) (parallel_generatedRoot_315.w_25_4 ctx) (parallel_generatedRoot_315.w_25_5 ctx) outputs) :
    parallel_generatedRoot_315.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_315.Facts  (ctx : parallel_generatedRoot_315Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_315.Facts_0 ctx outputs

theorem parallel_generatedRoot_315.facts  (ctx : parallel_generatedRoot_315Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_315.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_315.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_315.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_316.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_316.InputType)
  witness : parallel_generatedRoot_316.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_316Ctx" => (parallel_generatedRoot_316.Ctx backend hashModel params i_0)

def parallel_generatedRoot_316.w_1_0  (ctx : parallel_generatedRoot_316Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_316.w_3_0  (ctx : parallel_generatedRoot_316Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_316.w_5_0  (ctx : parallel_generatedRoot_316Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_316.w_7_0  (ctx : parallel_generatedRoot_316Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_316.w_9_0  (ctx : parallel_generatedRoot_316Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_316.w_10_0  (ctx : parallel_generatedRoot_316Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_316.w_11_0  (ctx : parallel_generatedRoot_316Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_316.w_12_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_316.w_13_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_316.w_14_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_316.w_15_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_316.w_16_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_316.w_17_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_316.w_0_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_316.w_2_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_316.w_4_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_316.w_6_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_316.w_8_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_316.w_18_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_316.w_18_1  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_316.w_18_2  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_316.w_18_3  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_316.w_18_4  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_316.w_18_5  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_316.w_18_6  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_316.w_18_7  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_316.w_18_8  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_316.w_18_9  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_316.w_19_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_316.w_20_0  (ctx : parallel_generatedRoot_316Ctx) : Int :=
  (MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])

def parallel_generatedRoot_316.w_21_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_316.w_22_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_316.w_23_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_316.w_23_concat_1  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_316.w_24_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_316.w_25_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_316.w_26_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_316.w_27_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_316.w_27_concat_1  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_316.w_28_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_316.w_29_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_316.w_30_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_316.w_31_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_316.w_32_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_316.w_32_concat_1  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_316.w_33_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_316.w_34_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_316.w_35_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_316.w_36_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_316.w_36_concat_1  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_316.w_37_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_316.w_38_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_316.w_39_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_316.w_40_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_316.w_41_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_316.w_37_0 ctx) (parallel_generatedRoot_316.w_40_0 ctx)

def parallel_generatedRoot_316.w_42_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_316.w_43_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_316.w_44_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_316.w_45_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_316.w_46_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_316.w_47_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_316.w_28_0 ctx) (parallel_generatedRoot_316.w_46_0 ctx)

def parallel_generatedRoot_316.w_48_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_316.w_49_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_18_2 ctx) (parallel_generatedRoot_316.w_48_0 ctx)

def parallel_generatedRoot_316.w_50_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_316.w_51_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_316.w_52_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_316.w_53_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_316.w_37_0 ctx) (parallel_generatedRoot_316.w_52_0 ctx)

def parallel_generatedRoot_316.w_54_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_316.w_55_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_316.w_56_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_316.w_57_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_316.w_58_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_316.w_59_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_316.w_28_0 ctx) (parallel_generatedRoot_316.w_58_0 ctx)

def parallel_generatedRoot_316.w_60_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_316.w_61_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_18_3 ctx) (parallel_generatedRoot_316.w_60_0 ctx)

def parallel_generatedRoot_316.w_62_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_316.w_63_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_18_4 ctx) (parallel_generatedRoot_316.w_62_0 ctx)

def parallel_generatedRoot_316.w_64_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_316.w_65_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_18_5 ctx) (parallel_generatedRoot_316.w_64_0 ctx)

def parallel_generatedRoot_316.w_66_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_316.w_67_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_316.w_68_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_66_0 ctx) (parallel_generatedRoot_316.w_67_0 ctx)

def parallel_generatedRoot_316.w_69_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_18_0 ctx) (parallel_generatedRoot_316.w_68_0 ctx)

def parallel_generatedRoot_316.w_70_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_316.w_71_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_316.w_72_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_70_0 ctx) (parallel_generatedRoot_316.w_71_0 ctx)

def parallel_generatedRoot_316.w_73_0  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_316.w_18_1 ctx) (parallel_generatedRoot_316.w_72_0 ctx)

def parallel_generatedRoot_316.outs  (ctx : parallel_generatedRoot_316Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_316.w_49_0 ctx), ((parallel_generatedRoot_316.w_61_0 ctx), ((parallel_generatedRoot_316.w_63_0 ctx), ((parallel_generatedRoot_316.w_65_0 ctx), ((parallel_generatedRoot_316.w_69_0 ctx), ((parallel_generatedRoot_316.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_316.Facts_0  (ctx : parallel_generatedRoot_316Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_316.w_1_0 ctx) (0) (parallel_generatedRoot_316.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_316.w_3_0 ctx) (0) (parallel_generatedRoot_316.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_316.w_5_0 ctx) (0) (parallel_generatedRoot_316.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_316.w_7_0 ctx) (0) (parallel_generatedRoot_316.w_8_0 ctx)
  t4 : 0 ≤ (127)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_316_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_316.w_9_0 ctx), ((parallel_generatedRoot_316.w_10_0 ctx), ((parallel_generatedRoot_316.w_11_0 ctx), ((parallel_generatedRoot_316.w_12_0 ctx), ((parallel_generatedRoot_316.w_13_0 ctx), ((parallel_generatedRoot_316.w_14_0 ctx), ((parallel_generatedRoot_316.w_15_0 ctx), ((parallel_generatedRoot_316.w_16_0 ctx), ((parallel_generatedRoot_316.w_17_0 ctx), ((parallel_generatedRoot_316.w_1_0 ctx), ((parallel_generatedRoot_316.w_3_0 ctx), ((parallel_generatedRoot_316.w_5_0 ctx), ((parallel_generatedRoot_316.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (127)) ((parallel_generatedRoot_316.w_0_0 ctx), ((parallel_generatedRoot_316.w_0_0 ctx), ((parallel_generatedRoot_316.w_0_0 ctx), ((parallel_generatedRoot_316.w_0_0 ctx), ((parallel_generatedRoot_316.w_0_0 ctx), ((parallel_generatedRoot_316.w_0_0 ctx), ((parallel_generatedRoot_316.w_2_0 ctx), ((parallel_generatedRoot_316.w_4_0 ctx), ((parallel_generatedRoot_316.w_6_0 ctx), ((parallel_generatedRoot_316.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_316.w_18_0 ctx), ((parallel_generatedRoot_316.w_18_1 ctx), ((parallel_generatedRoot_316.w_18_2 ctx), ((parallel_generatedRoot_316.w_18_3 ctx), ((parallel_generatedRoot_316.w_18_4 ctx), ((parallel_generatedRoot_316.w_18_5 ctx), ((parallel_generatedRoot_316.w_18_6 ctx), ((parallel_generatedRoot_316.w_18_7 ctx), ((parallel_generatedRoot_316.w_18_8 ctx), ((parallel_generatedRoot_316.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_316.w_9_0 ctx) (127) (parallel_generatedRoot_316.w_19_0 ctx)
  t7 : 0 ≤ (127)
  t8 : (127) < 128
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_316.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_316.w_10_0 ctx)) (parallel_generatedRoot_316.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_316.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_316.w_10_0 ctx)) (parallel_generatedRoot_316.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_19_0 ctx) (parallel_generatedRoot_316.w_21_0 ctx) (parallel_generatedRoot_316.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_23_concat_1 ctx) (parallel_generatedRoot_316.w_22_0 ctx) (parallel_generatedRoot_316.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_316.w_11_0 ctx) (127) (parallel_generatedRoot_316.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_316.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_316.w_10_0 ctx)) (parallel_generatedRoot_316.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_316.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_316.w_10_0 ctx)) (parallel_generatedRoot_316.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_24_0 ctx) (parallel_generatedRoot_316.w_25_0 ctx) (parallel_generatedRoot_316.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_27_concat_1 ctx) (parallel_generatedRoot_316.w_26_0 ctx) (parallel_generatedRoot_316.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_316.w_23_0 ctx) (parallel_generatedRoot_316.w_27_0 ctx) (parallel_generatedRoot_316.w_28_0 ctx)
  t19 : 0 ≤ (127)
  t20 : (127) < 128
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_316.w_12_0 ctx) (parallel_generatedRoot_316.w_29_0 ctx)
  t22 : 0 ≤ (127)
  t23 : (127) < 128
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_316.w_13_0 ctx) (parallel_generatedRoot_316.w_30_0 ctx)
  t25 : 0 ≤ (127)
  t26 : (127) < 128
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_316.w_14_0 ctx) (parallel_generatedRoot_316.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_29_0 ctx) (parallel_generatedRoot_316.w_30_0 ctx) (parallel_generatedRoot_316.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_32_concat_1 ctx) (parallel_generatedRoot_316.w_31_0 ctx) (parallel_generatedRoot_316.w_32_0 ctx)
  t30 : 0 ≤ (127)
  t31 : (127) < 128

theorem parallel_generatedRoot_316.facts_0  {ctx : parallel_generatedRoot_316Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_316.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_316.w_1_0 ctx) (parallel_generatedRoot_316.w_3_0 ctx) (parallel_generatedRoot_316.w_5_0 ctx) (parallel_generatedRoot_316.w_7_0 ctx) (parallel_generatedRoot_316.w_9_0 ctx) (parallel_generatedRoot_316.w_10_0 ctx) (parallel_generatedRoot_316.w_11_0 ctx) (parallel_generatedRoot_316.w_12_0 ctx) (parallel_generatedRoot_316.w_13_0 ctx) (parallel_generatedRoot_316.w_14_0 ctx) (parallel_generatedRoot_316.w_15_0 ctx) (parallel_generatedRoot_316.w_16_0 ctx) (parallel_generatedRoot_316.w_17_0 ctx) (parallel_generatedRoot_316.w_0_0 ctx) (parallel_generatedRoot_316.w_2_0 ctx) (parallel_generatedRoot_316.w_4_0 ctx) (parallel_generatedRoot_316.w_6_0 ctx) (parallel_generatedRoot_316.w_8_0 ctx) (parallel_generatedRoot_316.w_18_0 ctx) (parallel_generatedRoot_316.w_18_1 ctx) (parallel_generatedRoot_316.w_18_2 ctx) (parallel_generatedRoot_316.w_18_3 ctx) (parallel_generatedRoot_316.w_18_4 ctx) (parallel_generatedRoot_316.w_18_5 ctx) (parallel_generatedRoot_316.w_18_6 ctx) (parallel_generatedRoot_316.w_18_7 ctx) (parallel_generatedRoot_316.w_18_8 ctx) (parallel_generatedRoot_316.w_18_9 ctx) (parallel_generatedRoot_316.w_19_0 ctx) (parallel_generatedRoot_316.w_20_0 ctx) (parallel_generatedRoot_316.w_21_0 ctx) (parallel_generatedRoot_316.w_22_0 ctx) (parallel_generatedRoot_316.w_23_0 ctx) (parallel_generatedRoot_316.w_23_concat_1 ctx) (parallel_generatedRoot_316.w_24_0 ctx) (parallel_generatedRoot_316.w_25_0 ctx) (parallel_generatedRoot_316.w_26_0 ctx) (parallel_generatedRoot_316.w_27_0 ctx) (parallel_generatedRoot_316.w_27_concat_1 ctx) (parallel_generatedRoot_316.w_28_0 ctx) (parallel_generatedRoot_316.w_29_0 ctx) (parallel_generatedRoot_316.w_30_0 ctx) (parallel_generatedRoot_316.w_31_0 ctx) (parallel_generatedRoot_316.w_32_0 ctx) (parallel_generatedRoot_316.w_32_concat_1 ctx)) :
    parallel_generatedRoot_316.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_316.Facts_1  (ctx : parallel_generatedRoot_316Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_316.w_15_0 ctx) (parallel_generatedRoot_316.w_33_0 ctx)
  t33 : 0 ≤ (127)
  t34 : (127) < 128
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_316.w_16_0 ctx) (parallel_generatedRoot_316.w_34_0 ctx)
  t36 : 0 ≤ (127)
  t37 : (127) < 128
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_316.w_17_0 ctx) (parallel_generatedRoot_316.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_33_0 ctx) (parallel_generatedRoot_316.w_34_0 ctx) (parallel_generatedRoot_316.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_36_concat_1 ctx) (parallel_generatedRoot_316.w_35_0 ctx) (parallel_generatedRoot_316.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_316.w_32_0 ctx) (parallel_generatedRoot_316.w_36_0 ctx) (parallel_generatedRoot_316.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_18_6 ctx) (parallel_generatedRoot_316.w_18_8 ctx) (parallel_generatedRoot_316.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_316.w_38_0 ctx) (parallel_generatedRoot_316.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_316.w_39_0 ctx) (parallel_generatedRoot_316.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 270
  t50 : 270 ≤ 270
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_41_0 ctx) 1 2 0 270 (parallel_generatedRoot_316.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 270
  t57 : 270 ≤ 270
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_41_0 ctx) 2 3 0 270 (parallel_generatedRoot_316.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_42_0 ctx) (parallel_generatedRoot_316.w_43_0 ctx) (parallel_generatedRoot_316.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_316.w_44_0 ctx) (parallel_generatedRoot_316.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_316.w_45_0 ctx) (parallel_generatedRoot_316.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_316.facts_1  {ctx : parallel_generatedRoot_316Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_316.constraints_1 backend (parallel_generatedRoot_316.w_15_0 ctx) (parallel_generatedRoot_316.w_16_0 ctx) (parallel_generatedRoot_316.w_17_0 ctx) (parallel_generatedRoot_316.w_18_6 ctx) (parallel_generatedRoot_316.w_18_8 ctx) (parallel_generatedRoot_316.w_32_0 ctx) (parallel_generatedRoot_316.w_33_0 ctx) (parallel_generatedRoot_316.w_34_0 ctx) (parallel_generatedRoot_316.w_35_0 ctx) (parallel_generatedRoot_316.w_36_0 ctx) (parallel_generatedRoot_316.w_36_concat_1 ctx) (parallel_generatedRoot_316.w_37_0 ctx) (parallel_generatedRoot_316.w_38_0 ctx) (parallel_generatedRoot_316.w_39_0 ctx) (parallel_generatedRoot_316.w_40_0 ctx) (parallel_generatedRoot_316.w_41_0 ctx) (parallel_generatedRoot_316.w_42_0 ctx) (parallel_generatedRoot_316.w_43_0 ctx) (parallel_generatedRoot_316.w_44_0 ctx) (parallel_generatedRoot_316.w_45_0 ctx) (parallel_generatedRoot_316.w_46_0 ctx)) :
    parallel_generatedRoot_316.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_316.Facts_2  (ctx : parallel_generatedRoot_316Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 270
  t67 : 270 ≤ 270
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_47_0 ctx) 1 2 0 270 (parallel_generatedRoot_316.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_18_7 ctx) (parallel_generatedRoot_316.w_18_9 ctx) (parallel_generatedRoot_316.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_316.w_50_0 ctx) (parallel_generatedRoot_316.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_316.w_51_0 ctx) (parallel_generatedRoot_316.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 270
  t77 : 270 ≤ 270
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_53_0 ctx) 1 2 0 270 (parallel_generatedRoot_316.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 270
  t84 : 270 ≤ 270
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_53_0 ctx) 2 3 0 270 (parallel_generatedRoot_316.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_316.w_54_0 ctx) (parallel_generatedRoot_316.w_55_0 ctx) (parallel_generatedRoot_316.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_316.w_56_0 ctx) (parallel_generatedRoot_316.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_316.w_57_0 ctx) (parallel_generatedRoot_316.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 270
  t94 : 270 ≤ 270
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_59_0 ctx) 1 2 0 270 (parallel_generatedRoot_316.w_60_0 ctx)

theorem parallel_generatedRoot_316.facts_2  {ctx : parallel_generatedRoot_316Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_316.constraints_2 backend (parallel_generatedRoot_316.w_18_7 ctx) (parallel_generatedRoot_316.w_18_9 ctx) (parallel_generatedRoot_316.w_47_0 ctx) (parallel_generatedRoot_316.w_48_0 ctx) (parallel_generatedRoot_316.w_50_0 ctx) (parallel_generatedRoot_316.w_51_0 ctx) (parallel_generatedRoot_316.w_52_0 ctx) (parallel_generatedRoot_316.w_53_0 ctx) (parallel_generatedRoot_316.w_54_0 ctx) (parallel_generatedRoot_316.w_55_0 ctx) (parallel_generatedRoot_316.w_56_0 ctx) (parallel_generatedRoot_316.w_57_0 ctx) (parallel_generatedRoot_316.w_58_0 ctx) (parallel_generatedRoot_316.w_59_0 ctx) (parallel_generatedRoot_316.w_60_0 ctx)) :
    parallel_generatedRoot_316.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_316.Facts_3  (ctx : parallel_generatedRoot_316Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 270
  t101 : 270 ≤ 270
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_47_0 ctx) 2 3 0 270 (parallel_generatedRoot_316.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 270
  t108 : 270 ≤ 270
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_59_0 ctx) 2 3 0 270 (parallel_generatedRoot_316.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 270
  t115 : 270 ≤ 270
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_47_0 ctx) 0 1 0 270 (parallel_generatedRoot_316.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 270
  t122 : 270 ≤ 270
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_41_0 ctx) 0 1 0 270 (parallel_generatedRoot_316.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_316.facts_3  {ctx : parallel_generatedRoot_316Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_316.constraints_3 (parallel_generatedRoot_316.w_41_0 ctx) (parallel_generatedRoot_316.w_47_0 ctx) (parallel_generatedRoot_316.w_59_0 ctx) (parallel_generatedRoot_316.w_62_0 ctx) (parallel_generatedRoot_316.w_64_0 ctx) (parallel_generatedRoot_316.w_66_0 ctx) (parallel_generatedRoot_316.w_67_0 ctx)) :
    parallel_generatedRoot_316.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_316.Facts_4  (ctx : parallel_generatedRoot_316Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 270
  t129 : 270 ≤ 270
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_59_0 ctx) 0 1 0 270 (parallel_generatedRoot_316.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 270
  t136 : 270 ≤ 270
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_316.w_53_0 ctx) 0 1 0 270 (parallel_generatedRoot_316.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_316.outs ctx

theorem parallel_generatedRoot_316.facts_4  {ctx : parallel_generatedRoot_316Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_316.constraints_4 (parallel_generatedRoot_316.w_49_0 ctx) (parallel_generatedRoot_316.w_53_0 ctx) (parallel_generatedRoot_316.w_59_0 ctx) (parallel_generatedRoot_316.w_61_0 ctx) (parallel_generatedRoot_316.w_63_0 ctx) (parallel_generatedRoot_316.w_65_0 ctx) (parallel_generatedRoot_316.w_69_0 ctx) (parallel_generatedRoot_316.w_70_0 ctx) (parallel_generatedRoot_316.w_71_0 ctx) (parallel_generatedRoot_316.w_73_0 ctx) outputs) :
    parallel_generatedRoot_316.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_316.Facts  (ctx : parallel_generatedRoot_316Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_316.Facts_0 ctx
  c1 : parallel_generatedRoot_316.Facts_1 ctx
  c2 : parallel_generatedRoot_316.Facts_2 ctx
  c3 : parallel_generatedRoot_316.Facts_3 ctx
  c4 : parallel_generatedRoot_316.Facts_4 ctx outputs

theorem parallel_generatedRoot_316.facts  (ctx : parallel_generatedRoot_316Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_316.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_316.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_316.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_316.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_316.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_316.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_316.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_348.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (sequential_generatedRoot_348.InputType)
  witness : sequential_generatedRoot_348.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_348Ctx" => (sequential_generatedRoot_348.Ctx backend hashModel params i_0)

def sequential_generatedRoot_348.w_0_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.inputs.1

def sequential_generatedRoot_348.w_31_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.inputs.2.1

def sequential_generatedRoot_348.w_35_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_348.w_39_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_348.w_43_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_348.w_47_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_348.w_2_0  (ctx : sequential_generatedRoot_348Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_348.w_8_0  (ctx : sequential_generatedRoot_348Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_348.w_12_0  (ctx : sequential_generatedRoot_348Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_348.w_16_0  (ctx : sequential_generatedRoot_348Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_348.w_20_0  (ctx : sequential_generatedRoot_348Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_348.w_24_0  (ctx : sequential_generatedRoot_348Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_348.w_1_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_348.w_3_0  (ctx : sequential_generatedRoot_348Ctx) : Int :=
  (65536 + (65536 * (Int.ofNat i_0)))

def sequential_generatedRoot_348.w_4_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_348.w_5_0  (ctx : sequential_generatedRoot_348Ctx) : Int :=
  (65537 + (65536 * (Int.ofNat i_0)))

def sequential_generatedRoot_348.w_6_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_348.w_7_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_4_0 ctx) (sequential_generatedRoot_348.w_6_0 ctx)

def sequential_generatedRoot_348.w_9_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_348.w_10_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_348.w_11_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_9_0 ctx) (sequential_generatedRoot_348.w_10_0 ctx)

def sequential_generatedRoot_348.w_13_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_348.w_14_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_348.w_15_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_13_0 ctx) (sequential_generatedRoot_348.w_14_0 ctx)

def sequential_generatedRoot_348.w_17_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_348.w_18_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_348.w_19_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_17_0 ctx) (sequential_generatedRoot_348.w_18_0 ctx)

def sequential_generatedRoot_348.w_21_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_348.w_22_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def sequential_generatedRoot_348.w_23_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_21_0 ctx) (sequential_generatedRoot_348.w_22_0 ctx)

def sequential_generatedRoot_348.w_25_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_348.w_26_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def sequential_generatedRoot_348.w_27_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_25_0 ctx) (sequential_generatedRoot_348.w_26_0 ctx)

def sequential_generatedRoot_348.w_28_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_28_0

def sequential_generatedRoot_348.w_28_1  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_28_1

def sequential_generatedRoot_348.w_28_2  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_28_2

def sequential_generatedRoot_348.w_28_3  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_28_3

def sequential_generatedRoot_348.w_28_4  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_28_4

def sequential_generatedRoot_348.w_28_5  (ctx : sequential_generatedRoot_348Ctx) : mxxMat0 :=
  ctx.witness.w_28_5

def sequential_generatedRoot_348.w_29_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_348.w_30_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_0_0 ctx) (sequential_generatedRoot_348.w_29_0 ctx)

def sequential_generatedRoot_348.w_32_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_348.w_33_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_348.w_34_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_31_0 ctx) (sequential_generatedRoot_348.w_33_0 ctx)

def sequential_generatedRoot_348.w_36_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_348.w_37_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.witness.w_37_0

def sequential_generatedRoot_348.w_38_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_35_0 ctx) (sequential_generatedRoot_348.w_37_0 ctx)

def sequential_generatedRoot_348.w_40_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_348.w_41_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.witness.w_41_0

def sequential_generatedRoot_348.w_42_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_39_0 ctx) (sequential_generatedRoot_348.w_41_0 ctx)

def sequential_generatedRoot_348.w_44_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_348.w_45_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.witness.w_45_0

def sequential_generatedRoot_348.w_46_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_43_0 ctx) (sequential_generatedRoot_348.w_45_0 ctx)

def sequential_generatedRoot_348.w_48_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_348.w_49_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  ctx.witness.w_49_0

def sequential_generatedRoot_348.w_50_0  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_348.w_47_0 ctx) (sequential_generatedRoot_348.w_49_0 ctx)

def sequential_generatedRoot_348.outs  (ctx : sequential_generatedRoot_348Ctx) : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit :=
  ((sequential_generatedRoot_348.w_30_0 ctx), ((sequential_generatedRoot_348.w_34_0 ctx), ((sequential_generatedRoot_348.w_38_0 ctx), ((sequential_generatedRoot_348.w_42_0 ctx), ((sequential_generatedRoot_348.w_46_0 ctx), ((sequential_generatedRoot_348.w_50_0 ctx), ()))))))

structure sequential_generatedRoot_348.Facts_0  (ctx : sequential_generatedRoot_348Ctx) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 306
  h3_0 : 0 ≤ (sequential_generatedRoot_348.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_348.w_3_0 ctx) < 20054016
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_2_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_348.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_348.w_5_0 ctx) < 20054016
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_2_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_348.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_348.w_3_0 ctx) < 20054016
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_8_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_348.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_348.w_5_0 ctx) < 20054016
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_8_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_348.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_348.w_3_0 ctx) < 20054016
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_12_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_348.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_348.w_5_0 ctx) < 20054016
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_12_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_348.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_348.w_3_0 ctx) < 20054016
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_16_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_348.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_348.w_5_0 ctx) < 20054016
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_16_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_18_0 ctx)
  h3_0_8 : 0 ≤ (sequential_generatedRoot_348.w_3_0 ctx)
  h3_0_9 : (sequential_generatedRoot_348.w_3_0 ctx) < 20054016
  h21_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_20_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_21_0 ctx)
  h5_0_8 : 0 ≤ (sequential_generatedRoot_348.w_5_0 ctx)
  h5_0_9 : (sequential_generatedRoot_348.w_5_0 ctx) < 20054016
  h22_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_20_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_22_0 ctx)

theorem sequential_generatedRoot_348.facts_0  {ctx : sequential_generatedRoot_348Ctx} {outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit}
    (h : sequential_generatedRoot_348.constraints_0 i_0 (sequential_generatedRoot_348.w_2_0 ctx) (sequential_generatedRoot_348.w_8_0 ctx) (sequential_generatedRoot_348.w_12_0 ctx) (sequential_generatedRoot_348.w_16_0 ctx) (sequential_generatedRoot_348.w_20_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_4_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_6_0 ctx) (sequential_generatedRoot_348.w_9_0 ctx) (sequential_generatedRoot_348.w_10_0 ctx) (sequential_generatedRoot_348.w_13_0 ctx) (sequential_generatedRoot_348.w_14_0 ctx) (sequential_generatedRoot_348.w_17_0 ctx) (sequential_generatedRoot_348.w_18_0 ctx) (sequential_generatedRoot_348.w_21_0 ctx) (sequential_generatedRoot_348.w_22_0 ctx)) :
    sequential_generatedRoot_348.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_348.Facts_1  (ctx : sequential_generatedRoot_348Ctx) (outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit) : Prop where
  h3_0_10 : 0 ≤ (sequential_generatedRoot_348.w_3_0 ctx)
  h3_0_11 : (sequential_generatedRoot_348.w_3_0 ctx) < 20054016
  h25_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_24_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_25_0 ctx)
  h5_0_10 : 0 ≤ (sequential_generatedRoot_348.w_5_0 ctx)
  h5_0_11 : (sequential_generatedRoot_348.w_5_0 ctx) < 20054016
  h26_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_348.w_24_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_26_0 ctx)
  t38 : 0 ≤ (65534)
  h28_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_145 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((sequential_generatedRoot_348.w_2_0 ctx), ((sequential_generatedRoot_348.w_8_0 ctx), ((sequential_generatedRoot_348.w_12_0 ctx), ((sequential_generatedRoot_348.w_16_0 ctx), ((sequential_generatedRoot_348.w_20_0 ctx), ((sequential_generatedRoot_348.w_24_0 ctx), ())))))))))))) next) (Int.toNat (65534)) ((sequential_generatedRoot_348.w_7_0 ctx), ((sequential_generatedRoot_348.w_11_0 ctx), ((sequential_generatedRoot_348.w_15_0 ctx), ((sequential_generatedRoot_348.w_19_0 ctx), ((sequential_generatedRoot_348.w_23_0 ctx), ((sequential_generatedRoot_348.w_27_0 ctx), ())))))) ((sequential_generatedRoot_348.w_28_0 ctx), ((sequential_generatedRoot_348.w_28_1 ctx), ((sequential_generatedRoot_348.w_28_2 ctx), ((sequential_generatedRoot_348.w_28_3 ctx), ((sequential_generatedRoot_348.w_28_4 ctx), ((sequential_generatedRoot_348.w_28_5 ctx), ()))))))
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_348.w_1_0 ctx) (sequential_generatedRoot_348.w_28_0 ctx) (sequential_generatedRoot_348.w_29_0 ctx)
  t41 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t42 : ((1 + (Int.ofNat i_0))) < 306
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_348.w_32_0 ctx) (sequential_generatedRoot_348.w_28_1 ctx) (sequential_generatedRoot_348.w_33_0 ctx)
  t44 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t45 : ((1 + (Int.ofNat i_0))) < 306
  h37_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_348.w_36_0 ctx) (sequential_generatedRoot_348.w_28_2 ctx) (sequential_generatedRoot_348.w_37_0 ctx)
  t47 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t48 : ((1 + (Int.ofNat i_0))) < 306
  h41_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_348.w_40_0 ctx) (sequential_generatedRoot_348.w_28_3 ctx) (sequential_generatedRoot_348.w_41_0 ctx)
  t50 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t51 : ((1 + (Int.ofNat i_0))) < 306
  h45_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_348.w_44_0 ctx) (sequential_generatedRoot_348.w_28_4 ctx) (sequential_generatedRoot_348.w_45_0 ctx)
  t53 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t54 : ((1 + (Int.ofNat i_0))) < 306
  h49_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_348.w_48_0 ctx) (sequential_generatedRoot_348.w_28_5 ctx) (sequential_generatedRoot_348.w_49_0 ctx)
  hout : outputs = sequential_generatedRoot_348.outs ctx

theorem sequential_generatedRoot_348.facts_1  {ctx : sequential_generatedRoot_348Ctx} {outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit}
    (h : sequential_generatedRoot_348.constraints_1 backend hashModel params i_0 (sequential_generatedRoot_348.w_2_0 ctx) (sequential_generatedRoot_348.w_8_0 ctx) (sequential_generatedRoot_348.w_12_0 ctx) (sequential_generatedRoot_348.w_16_0 ctx) (sequential_generatedRoot_348.w_20_0 ctx) (sequential_generatedRoot_348.w_24_0 ctx) (sequential_generatedRoot_348.w_1_0 ctx) (sequential_generatedRoot_348.w_3_0 ctx) (sequential_generatedRoot_348.w_5_0 ctx) (sequential_generatedRoot_348.w_7_0 ctx) (sequential_generatedRoot_348.w_11_0 ctx) (sequential_generatedRoot_348.w_15_0 ctx) (sequential_generatedRoot_348.w_19_0 ctx) (sequential_generatedRoot_348.w_23_0 ctx) (sequential_generatedRoot_348.w_25_0 ctx) (sequential_generatedRoot_348.w_26_0 ctx) (sequential_generatedRoot_348.w_27_0 ctx) (sequential_generatedRoot_348.w_28_0 ctx) (sequential_generatedRoot_348.w_28_1 ctx) (sequential_generatedRoot_348.w_28_2 ctx) (sequential_generatedRoot_348.w_28_3 ctx) (sequential_generatedRoot_348.w_28_4 ctx) (sequential_generatedRoot_348.w_28_5 ctx) (sequential_generatedRoot_348.w_29_0 ctx) (sequential_generatedRoot_348.w_30_0 ctx) (sequential_generatedRoot_348.w_32_0 ctx) (sequential_generatedRoot_348.w_33_0 ctx) (sequential_generatedRoot_348.w_34_0 ctx) (sequential_generatedRoot_348.w_36_0 ctx) (sequential_generatedRoot_348.w_37_0 ctx) (sequential_generatedRoot_348.w_38_0 ctx) (sequential_generatedRoot_348.w_40_0 ctx) (sequential_generatedRoot_348.w_41_0 ctx) (sequential_generatedRoot_348.w_42_0 ctx) (sequential_generatedRoot_348.w_44_0 ctx) (sequential_generatedRoot_348.w_45_0 ctx) (sequential_generatedRoot_348.w_46_0 ctx) (sequential_generatedRoot_348.w_48_0 ctx) (sequential_generatedRoot_348.w_49_0 ctx) (sequential_generatedRoot_348.w_50_0 ctx) outputs) :
    sequential_generatedRoot_348.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_348.Facts  (ctx : sequential_generatedRoot_348Ctx) (outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit) : Prop where
  c0 : sequential_generatedRoot_348.Facts_0 ctx
  c1 : sequential_generatedRoot_348.Facts_1 ctx outputs

theorem sequential_generatedRoot_348.facts  (ctx : sequential_generatedRoot_348Ctx) {outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit}
    (h : sequential_generatedRoot_348.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_348.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_348.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_348.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel params)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  h3_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (generatedRoot.w_5_0 ctx) (0) (generatedRoot.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (generatedRoot.w_7_0 ctx) (0) (generatedRoot.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (generatedRoot.w_9_0 ctx) (0) (generatedRoot.w_10_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (generatedRoot.w_11_0 ctx) (0) (generatedRoot.w_12_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (generatedRoot.w_16_0 ctx) (1024) (generatedRoot.w_17_0 ctx)
  h20_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_20_0 ctx)
  h23_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_23_0 ctx)
  h22_0 : MxxRuntime.concatRows (generatedRoot.w_19_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_26_concat_1 ctx)
  h26_0 : MxxRuntime.concatRows (generatedRoot.w_26_concat_1 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx)
  h28_0 : MxxRuntime.familyGetStatic (generatedRoot.w_27_0 ctx) (1024) (generatedRoot.w_28_0 ctx)
  h31_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_31_0 ctx)
  h34_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_34_0 ctx)
  h33_0 : MxxRuntime.concatRows (generatedRoot.w_30_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_37_concat_1 ctx)
  h37_0 : MxxRuntime.concatRows (generatedRoot.w_37_concat_1 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx)
  h38_0 : MxxRuntime.concatColumns (generatedRoot.w_26_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx)
  h39_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_39_0 ctx)
  h40_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_40_0 ctx)
  h41_0 : MxxRuntime.concatRows (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx)
  h42_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx)
  h43_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx)
  t21 : 0 ≤ 0
  t22 : 0 < 1
  t23 : 1 ≤ 3
  t24 : 0 ≤ 0
  t25 : 0 < 270
  t26 : 270 ≤ 270
  h45_0 : MxxRuntime.sliceMatrix (generatedRoot.w_44_0 ctx) 0 1 0 270 (generatedRoot.w_45_0 ctx)
  h48_0 : MxxRuntime.familyGetStatic (generatedRoot.w_47_0 ctx) (1024) (generatedRoot.w_48_0 ctx)
  t29 : 0 ≤ 1
  t30 : 1 < 2
  t31 : 2 ≤ 3

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_26_concat_1 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_37_concat_1 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_48_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  t32 : 0 ≤ 0
  t33 : 0 < 270
  t34 : 270 ≤ 270
  h53_0 : MxxRuntime.sliceMatrix (generatedRoot.w_44_0 ctx) 1 2 0 270 (generatedRoot.w_53_0 ctx)
  t36 : 0 ≤ 2
  t37 : 2 < 3
  t38 : 3 ≤ 3
  t39 : 0 ≤ 0
  t40 : 0 < 270
  t41 : 270 ≤ 270
  h54_0 : MxxRuntime.sliceMatrix (generatedRoot.w_44_0 ctx) 2 3 0 270 (generatedRoot.w_54_0 ctx)
  h53_0_1 : MxxRuntime.concatRows (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_concat_1 ctx)
  h55_0 : MxxRuntime.concatRows (generatedRoot.w_55_concat_1 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx)
  h56_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_56_0 ctx)
  h57_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_57_0 ctx)
  h58_0 : MxxRuntime.concatRows (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx)
  h59_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx)
  h60_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx)
  t50 : 0 ≤ 0
  t51 : 0 < 1
  t52 : 1 ≤ 3
  t53 : 0 ≤ 0
  t54 : 0 < 270
  t55 : 270 ≤ 270
  h62_0 : MxxRuntime.sliceMatrix (generatedRoot.w_61_0 ctx) 0 1 0 270 (generatedRoot.w_62_0 ctx)
  t57 : 0 ≤ 1
  t58 : 1 < 2
  t59 : 2 ≤ 3
  t60 : 0 ≤ 0
  t61 : 0 < 270
  t62 : 270 ≤ 270
  h66_0 : MxxRuntime.sliceMatrix (generatedRoot.w_61_0 ctx) 1 2 0 270 (generatedRoot.w_66_0 ctx)

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_14_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_55_concat_1 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_66_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
  t64 : 0 ≤ 2
  t65 : 2 < 3
  t66 : 3 ≤ 3
  t67 : 0 ≤ 0
  t68 : 0 < 270
  t69 : 270 ≤ 270
  h67_0 : MxxRuntime.sliceMatrix (generatedRoot.w_61_0 ctx) 2 3 0 270 (generatedRoot.w_67_0 ctx)
  h66_0_1 : MxxRuntime.concatRows (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_68_concat_1 ctx)
  h68_0 : MxxRuntime.concatRows (generatedRoot.w_68_concat_1 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx)
  h69_0 : MxxRuntime.concatColumns (generatedRoot.w_55_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_69_0 ctx)
  h71_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 52, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat0), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat1), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_70_0 ctx)) (generatedRoot.w_71_0 ctx)
  h73_0 : MxxRuntime.familyGetStatic (generatedRoot.w_72_0 ctx) (0) (generatedRoot.w_73_0 ctx)
  h74_0 : MxxRuntime.concatRows (generatedRoot.w_71_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx)
  h75_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx)
  h76_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx)
  t79 : 0 ≤ 0
  t80 : 0 < 1
  t81 : 1 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 270
  t84 : 270 ≤ 270
  h78_0 : MxxRuntime.sliceMatrix (generatedRoot.w_77_0 ctx) 0 1 0 270 (generatedRoot.w_78_0 ctx)
  t86 : 0 ≤ 1
  t87 : 1 < 2
  t88 : 2 ≤ 3
  t89 : 0 ≤ 0
  t90 : 0 < 270
  t91 : 270 ≤ 270
  h79_0 : MxxRuntime.sliceMatrix (generatedRoot.w_77_0 ctx) 1 2 0 270 (generatedRoot.w_79_0 ctx)
  t93 : 0 ≤ 2
  t94 : 2 < 3
  t95 : 3 ≤ 3

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend hashModel (generatedRoot.w_70_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_68_concat_1 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
  t96 : 0 ≤ 0
  t97 : 0 < 270
  t98 : 270 ≤ 270
  h80_0 : MxxRuntime.sliceMatrix (generatedRoot.w_77_0 ctx) 2 3 0 270 (generatedRoot.w_80_0 ctx)
  h81_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 52, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat0), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat1), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_70_0 ctx)) (generatedRoot.w_81_0 ctx)
  h83_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_0 ctx) (0) (generatedRoot.w_83_0 ctx)
  h84_0 : MxxRuntime.concatRows (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx)
  h85_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx)
  h86_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx)
  t105 : 0 ≤ 0
  t106 : 0 < 1
  t107 : 1 ≤ 3
  t108 : 0 ≤ 0
  t109 : 0 < 270
  t110 : 270 ≤ 270
  h88_0 : MxxRuntime.sliceMatrix (generatedRoot.w_87_0 ctx) 0 1 0 270 (generatedRoot.w_88_0 ctx)
  t112 : 0 ≤ 1
  t113 : 1 < 2
  t114 : 2 ≤ 3
  t115 : 0 ≤ 0
  t116 : 0 < 270
  t117 : 270 ≤ 270
  h89_0 : MxxRuntime.sliceMatrix (generatedRoot.w_87_0 ctx) 1 2 0 270 (generatedRoot.w_89_0 ctx)
  t119 : 0 ≤ 2
  t120 : 2 < 3
  t121 : 3 ≤ 3
  t122 : 0 ≤ 0
  t123 : 0 < 270
  t124 : 270 ≤ 270
  h90_0 : MxxRuntime.sliceMatrix (generatedRoot.w_87_0 ctx) 2 3 0 270 (generatedRoot.w_90_0 ctx)
  t126 : 0 ≤ (3)
  h91_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_91 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_78_0 ctx), ((generatedRoot.w_79_0 ctx), ((generatedRoot.w_80_0 ctx), ((generatedRoot.w_88_0 ctx), ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_11_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (3)) ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_12_0 ctx), ())))))))))) ((generatedRoot.w_91_0 ctx), ((generatedRoot.w_91_1 ctx), ((generatedRoot.w_91_2 ctx), ((generatedRoot.w_91_3 ctx), ((generatedRoot.w_91_4 ctx), ((generatedRoot.w_91_5 ctx), ((generatedRoot.w_91_6 ctx), ((generatedRoot.w_91_7 ctx), ((generatedRoot.w_91_8 ctx), ((generatedRoot.w_91_9 ctx), ()))))))))))

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_3 backend hashModel params (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_91_1 ctx) (generatedRoot.w_91_2 ctx) (generatedRoot.w_91_3 ctx) (generatedRoot.w_91_4 ctx) (generatedRoot.w_91_5 ctx) (generatedRoot.w_91_6 ctx) (generatedRoot.w_91_7 ctx) (generatedRoot.w_91_8 ctx) (generatedRoot.w_91_9 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4  (ctx : generatedRootCtx) : Prop where
  h92_0 : MxxRuntime.familyGetStatic (generatedRoot.w_13_0 ctx) (3) (generatedRoot.w_92_0 ctx)
  t129 : 0 ≤ (3)
  t130 : (3) < 4
  h94_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_93_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_94_0 ctx)
  h95_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_93_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_95_0 ctx)
  h94_0_1 : MxxRuntime.concatRows (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_concat_1 ctx)
  h96_0 : MxxRuntime.concatRows (generatedRoot.w_96_concat_1 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx)
  h97_0 : MxxRuntime.familyGetStatic (generatedRoot.w_15_0 ctx) (3) (generatedRoot.w_97_0 ctx)
  h98_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_93_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_98_0 ctx)
  h99_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_93_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_99_0 ctx)
  h98_0_1 : MxxRuntime.concatRows (generatedRoot.w_97_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_concat_1 ctx)
  h100_0 : MxxRuntime.concatRows (generatedRoot.w_100_concat_1 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx)
  h101_0 : MxxRuntime.concatColumns (generatedRoot.w_96_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx)
  t141 : 0 ≤ (3)
  t142 : (3) < 4
  h102_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_78_0 ctx) (generatedRoot.w_102_0 ctx)
  t144 : 0 ≤ (3)
  t145 : (3) < 4
  h103_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_79_0 ctx) (generatedRoot.w_103_0 ctx)
  t147 : 0 ≤ (3)
  t148 : (3) < 4
  h104_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_80_0 ctx) (generatedRoot.w_104_0 ctx)
  h103_0_1 : MxxRuntime.concatRows (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_concat_1 ctx)
  h105_0 : MxxRuntime.concatRows (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx)
  t152 : 0 ≤ (3)
  t153 : (3) < 4
  h106_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_88_0 ctx) (generatedRoot.w_106_0 ctx)
  t155 : 0 ≤ (3)
  t156 : (3) < 4
  h107_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_89_0 ctx) (generatedRoot.w_107_0 ctx)
  t158 : 0 ≤ (3)
  t159 : (3) < 4

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_4 hashModel (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_96_concat_1 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_100_concat_1 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
  h108_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_90_0 ctx) (generatedRoot.w_108_0 ctx)
  h107_0_1 : MxxRuntime.concatRows (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_concat_1 ctx)
  h109_0 : MxxRuntime.concatRows (generatedRoot.w_109_concat_1 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx)
  h110_0 : MxxRuntime.concatColumns (generatedRoot.w_105_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx)
  h111_0 : MxxRuntime.concatRows (generatedRoot.w_91_7 ctx) (generatedRoot.w_91_9 ctx) (generatedRoot.w_111_0 ctx)
  h112_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx)
  h113_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx)
  t167 : 0 ≤ 1
  t168 : 1 < 2
  t169 : 2 ≤ 3
  t170 : 0 ≤ 0
  t171 : 0 < 270
  t172 : 270 ≤ 270
  h115_0 : MxxRuntime.sliceMatrix (generatedRoot.w_114_0 ctx) 1 2 0 270 (generatedRoot.w_115_0 ctx)
  t174 : 0 ≤ 2
  t175 : 2 < 3
  t176 : 3 ≤ 3
  t177 : 0 ≤ 0
  t178 : 0 < 270
  t179 : 270 ≤ 270
  h116_0 : MxxRuntime.sliceMatrix (generatedRoot.w_114_0 ctx) 2 3 0 270 (generatedRoot.w_116_0 ctx)
  h117_0 : MxxRuntime.concatRows (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_117_0 ctx)
  h118_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_117_0 ctx) (generatedRoot.w_118_0 ctx)
  h119_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_118_0 ctx) (generatedRoot.w_119_0 ctx)
  t184 : 0 ≤ 0
  t185 : 0 < 1
  t186 : 1 ≤ 3
  t187 : 0 ≤ 0
  t188 : 0 < 270
  t189 : 270 ≤ 270
  h121_0 : MxxRuntime.sliceMatrix (generatedRoot.w_120_0 ctx) 0 1 0 270 (generatedRoot.w_121_0 ctx)
  t191 : 0 ≤ 0

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_5 backend (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_7 ctx) (generatedRoot.w_91_9 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_109_concat_1 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_121_0 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
  t192 : 0 < 1
  t193 : 1 ≤ 3
  t194 : 0 ≤ 0
  t195 : 0 < 270
  t196 : 270 ≤ 270
  h122_0 : MxxRuntime.sliceMatrix (generatedRoot.w_114_0 ctx) 0 1 0 270 (generatedRoot.w_122_0 ctx)
  h126_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx)
  h129_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)
  h130_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx)
  t201 : 0 ≤ (0)
  t202 : (0) < 306
  h174_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_174_0 ctx)
  h175_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_175_0 ctx)
  h177_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_177_0 ctx)
  h178_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_14_0 ctx)) (generatedRoot.w_178_0 ctx)
  t207 : (2607022080) = 2607022080
  h182_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_173_0 ctx) i) ∧ ((generatedRoot.w_173_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_173_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_173_0 ctx) i) ∧ ((generatedRoot.w_173_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_173_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_173_0 ctx) i) ∧ ((generatedRoot.w_173_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_173_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_182 backend hashModel params i (((generatedRoot.w_173_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_174_0 ctx), ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_176_0 ctx), (w_47_0_gathered, ((generatedRoot.w_177_0 ctx), ((generatedRoot.w_178_0 ctx), ((generatedRoot.w_179_0 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_180_0 ctx) i), (((generatedRoot.w_181_0 ctx) i), ())))))))))))))) (((generatedRoot.w_182_0 ctx) i), (((generatedRoot.w_182_1 ctx) i), (((generatedRoot.w_182_2 ctx) i), (((generatedRoot.w_182_3 ctx) i), (((generatedRoot.w_182_4 ctx) i), (((generatedRoot.w_182_5 ctx) i), ())))))))
  t209 : (20054016) = 20054016
  h183_5 : (∀ i : Fin 20054016, parallel_generatedRoot_183 backend hashModel params i ((generatedRoot.w_182_0 ctx), ((generatedRoot.w_182_1 ctx), ((generatedRoot.w_182_2 ctx), ((generatedRoot.w_182_3 ctx), ((generatedRoot.w_182_4 ctx), ((generatedRoot.w_182_5 ctx), ())))))) (((generatedRoot.w_183_0 ctx) i), (((generatedRoot.w_183_1 ctx) i), (((generatedRoot.w_183_2 ctx) i), (((generatedRoot.w_183_3 ctx) i), (((generatedRoot.w_183_4 ctx) i), (((generatedRoot.w_183_5 ctx) i), ())))))))
  t211 : (2607022080) = 2607022080
  h186_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_172_0 ctx) i) ∧ ((generatedRoot.w_172_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_172_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_172_0 ctx) i) ∧ ((generatedRoot.w_172_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_172_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_172_0 ctx) i) ∧ ((generatedRoot.w_172_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_172_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_186 backend hashModel params i (((generatedRoot.w_172_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_183_0 ctx), ((generatedRoot.w_183_2 ctx), ((generatedRoot.w_183_4 ctx), (w_47_0_gathered, ((generatedRoot.w_183_1 ctx), ((generatedRoot.w_183_3 ctx), ((generatedRoot.w_183_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_184_0 ctx) i), (((generatedRoot.w_185_0 ctx) i), ())))))))))))))) (((generatedRoot.w_186_0 ctx) i), (((generatedRoot.w_186_1 ctx) i), (((generatedRoot.w_186_2 ctx) i), (((generatedRoot.w_186_3 ctx) i), (((generatedRoot.w_186_4 ctx) i), (((generatedRoot.w_186_5 ctx) i), ())))))))
  t213 : (20054016) = 20054016
  h187_5 : (∀ i : Fin 20054016, parallel_generatedRoot_187 backend hashModel params i ((generatedRoot.w_186_0 ctx), ((generatedRoot.w_186_1 ctx), ((generatedRoot.w_186_2 ctx), ((generatedRoot.w_186_3 ctx), ((generatedRoot.w_186_4 ctx), ((generatedRoot.w_186_5 ctx), ())))))) (((generatedRoot.w_187_0 ctx) i), (((generatedRoot.w_187_1 ctx) i), (((generatedRoot.w_187_2 ctx) i), (((generatedRoot.w_187_3 ctx) i), (((generatedRoot.w_187_4 ctx) i), (((generatedRoot.w_187_5 ctx) i), ())))))))
  t215 : (2607022080) = 2607022080
  h190_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_171_0 ctx) i) ∧ ((generatedRoot.w_171_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_171_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_171_0 ctx) i) ∧ ((generatedRoot.w_171_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_171_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_171_0 ctx) i) ∧ ((generatedRoot.w_171_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_171_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_190 backend hashModel params i (((generatedRoot.w_171_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_187_0 ctx), ((generatedRoot.w_187_2 ctx), ((generatedRoot.w_187_4 ctx), (w_47_0_gathered, ((generatedRoot.w_187_1 ctx), ((generatedRoot.w_187_3 ctx), ((generatedRoot.w_187_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_188_0 ctx) i), (((generatedRoot.w_189_0 ctx) i), ())))))))))))))) (((generatedRoot.w_190_0 ctx) i), (((generatedRoot.w_190_1 ctx) i), (((generatedRoot.w_190_2 ctx) i), (((generatedRoot.w_190_3 ctx) i), (((generatedRoot.w_190_4 ctx) i), (((generatedRoot.w_190_5 ctx) i), ())))))))
  t217 : (20054016) = 20054016
  h191_5 : (∀ i : Fin 20054016, parallel_generatedRoot_191 backend hashModel params i ((generatedRoot.w_190_0 ctx), ((generatedRoot.w_190_1 ctx), ((generatedRoot.w_190_2 ctx), ((generatedRoot.w_190_3 ctx), ((generatedRoot.w_190_4 ctx), ((generatedRoot.w_190_5 ctx), ())))))) (((generatedRoot.w_191_0 ctx) i), (((generatedRoot.w_191_1 ctx) i), (((generatedRoot.w_191_2 ctx) i), (((generatedRoot.w_191_3 ctx) i), (((generatedRoot.w_191_4 ctx) i), (((generatedRoot.w_191_5 ctx) i), ())))))))
  t219 : (2607022080) = 2607022080
  h194_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_170_0 ctx) i) ∧ ((generatedRoot.w_170_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_170_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_170_0 ctx) i) ∧ ((generatedRoot.w_170_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_170_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_170_0 ctx) i) ∧ ((generatedRoot.w_170_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_170_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_194 backend hashModel params i (((generatedRoot.w_170_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_191_0 ctx), ((generatedRoot.w_191_2 ctx), ((generatedRoot.w_191_4 ctx), (w_47_0_gathered, ((generatedRoot.w_191_1 ctx), ((generatedRoot.w_191_3 ctx), ((generatedRoot.w_191_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_192_0 ctx) i), (((generatedRoot.w_193_0 ctx) i), ())))))))))))))) (((generatedRoot.w_194_0 ctx) i), (((generatedRoot.w_194_1 ctx) i), (((generatedRoot.w_194_2 ctx) i), (((generatedRoot.w_194_3 ctx) i), (((generatedRoot.w_194_4 ctx) i), (((generatedRoot.w_194_5 ctx) i), ())))))))
  t221 : (20054016) = 20054016
  h195_5 : (∀ i : Fin 20054016, parallel_generatedRoot_195 backend hashModel params i ((generatedRoot.w_194_0 ctx), ((generatedRoot.w_194_1 ctx), ((generatedRoot.w_194_2 ctx), ((generatedRoot.w_194_3 ctx), ((generatedRoot.w_194_4 ctx), ((generatedRoot.w_194_5 ctx), ())))))) (((generatedRoot.w_195_0 ctx) i), (((generatedRoot.w_195_1 ctx) i), (((generatedRoot.w_195_2 ctx) i), (((generatedRoot.w_195_3 ctx) i), (((generatedRoot.w_195_4 ctx) i), (((generatedRoot.w_195_5 ctx) i), ())))))))
  t223 : (2607022080) = 2607022080

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_6 backend hashModel params (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_182_1 ctx) (generatedRoot.w_182_2 ctx) (generatedRoot.w_182_3 ctx) (generatedRoot.w_182_4 ctx) (generatedRoot.w_182_5 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_183_1 ctx) (generatedRoot.w_183_2 ctx) (generatedRoot.w_183_3 ctx) (generatedRoot.w_183_4 ctx) (generatedRoot.w_183_5 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_186_1 ctx) (generatedRoot.w_186_2 ctx) (generatedRoot.w_186_3 ctx) (generatedRoot.w_186_4 ctx) (generatedRoot.w_186_5 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_187_1 ctx) (generatedRoot.w_187_2 ctx) (generatedRoot.w_187_3 ctx) (generatedRoot.w_187_4 ctx) (generatedRoot.w_187_5 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_191_1 ctx) (generatedRoot.w_191_2 ctx) (generatedRoot.w_191_3 ctx) (generatedRoot.w_191_4 ctx) (generatedRoot.w_191_5 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_194_1 ctx) (generatedRoot.w_194_2 ctx) (generatedRoot.w_194_3 ctx) (generatedRoot.w_194_4 ctx) (generatedRoot.w_194_5 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_195_1 ctx) (generatedRoot.w_195_2 ctx) (generatedRoot.w_195_3 ctx) (generatedRoot.w_195_4 ctx) (generatedRoot.w_195_5 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) : Prop where
  h198_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_169_0 ctx) i) ∧ ((generatedRoot.w_169_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_169_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_169_0 ctx) i) ∧ ((generatedRoot.w_169_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_169_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_169_0 ctx) i) ∧ ((generatedRoot.w_169_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_169_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_198 backend hashModel params i (((generatedRoot.w_169_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_195_0 ctx), ((generatedRoot.w_195_2 ctx), ((generatedRoot.w_195_4 ctx), (w_47_0_gathered, ((generatedRoot.w_195_1 ctx), ((generatedRoot.w_195_3 ctx), ((generatedRoot.w_195_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_196_0 ctx) i), (((generatedRoot.w_197_0 ctx) i), ())))))))))))))) (((generatedRoot.w_198_0 ctx) i), (((generatedRoot.w_198_1 ctx) i), (((generatedRoot.w_198_2 ctx) i), (((generatedRoot.w_198_3 ctx) i), (((generatedRoot.w_198_4 ctx) i), (((generatedRoot.w_198_5 ctx) i), ())))))))
  t225 : (20054016) = 20054016
  h199_5 : (∀ i : Fin 20054016, parallel_generatedRoot_199 backend hashModel params i ((generatedRoot.w_198_0 ctx), ((generatedRoot.w_198_1 ctx), ((generatedRoot.w_198_2 ctx), ((generatedRoot.w_198_3 ctx), ((generatedRoot.w_198_4 ctx), ((generatedRoot.w_198_5 ctx), ())))))) (((generatedRoot.w_199_0 ctx) i), (((generatedRoot.w_199_1 ctx) i), (((generatedRoot.w_199_2 ctx) i), (((generatedRoot.w_199_3 ctx) i), (((generatedRoot.w_199_4 ctx) i), (((generatedRoot.w_199_5 ctx) i), ())))))))
  t227 : (2607022080) = 2607022080
  h202_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_168_0 ctx) i) ∧ ((generatedRoot.w_168_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_168_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_168_0 ctx) i) ∧ ((generatedRoot.w_168_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_168_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_168_0 ctx) i) ∧ ((generatedRoot.w_168_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_168_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_202 backend hashModel params i (((generatedRoot.w_168_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_199_0 ctx), ((generatedRoot.w_199_2 ctx), ((generatedRoot.w_199_4 ctx), (w_47_0_gathered, ((generatedRoot.w_199_1 ctx), ((generatedRoot.w_199_3 ctx), ((generatedRoot.w_199_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_200_0 ctx) i), (((generatedRoot.w_201_0 ctx) i), ())))))))))))))) (((generatedRoot.w_202_0 ctx) i), (((generatedRoot.w_202_1 ctx) i), (((generatedRoot.w_202_2 ctx) i), (((generatedRoot.w_202_3 ctx) i), (((generatedRoot.w_202_4 ctx) i), (((generatedRoot.w_202_5 ctx) i), ())))))))
  t229 : (20054016) = 20054016
  h203_5 : (∀ i : Fin 20054016, parallel_generatedRoot_203 backend hashModel params i ((generatedRoot.w_202_0 ctx), ((generatedRoot.w_202_1 ctx), ((generatedRoot.w_202_2 ctx), ((generatedRoot.w_202_3 ctx), ((generatedRoot.w_202_4 ctx), ((generatedRoot.w_202_5 ctx), ())))))) (((generatedRoot.w_203_0 ctx) i), (((generatedRoot.w_203_1 ctx) i), (((generatedRoot.w_203_2 ctx) i), (((generatedRoot.w_203_3 ctx) i), (((generatedRoot.w_203_4 ctx) i), (((generatedRoot.w_203_5 ctx) i), ())))))))
  t231 : (2607022080) = 2607022080
  h206_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_167_0 ctx) i) ∧ ((generatedRoot.w_167_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_167_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_167_0 ctx) i) ∧ ((generatedRoot.w_167_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_167_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_167_0 ctx) i) ∧ ((generatedRoot.w_167_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_167_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_206 backend hashModel params i (((generatedRoot.w_167_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_203_0 ctx), ((generatedRoot.w_203_2 ctx), ((generatedRoot.w_203_4 ctx), (w_47_0_gathered, ((generatedRoot.w_203_1 ctx), ((generatedRoot.w_203_3 ctx), ((generatedRoot.w_203_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_204_0 ctx) i), (((generatedRoot.w_205_0 ctx) i), ())))))))))))))) (((generatedRoot.w_206_0 ctx) i), (((generatedRoot.w_206_1 ctx) i), (((generatedRoot.w_206_2 ctx) i), (((generatedRoot.w_206_3 ctx) i), (((generatedRoot.w_206_4 ctx) i), (((generatedRoot.w_206_5 ctx) i), ())))))))
  t233 : (20054016) = 20054016
  h207_5 : (∀ i : Fin 20054016, parallel_generatedRoot_207 backend hashModel params i ((generatedRoot.w_206_0 ctx), ((generatedRoot.w_206_1 ctx), ((generatedRoot.w_206_2 ctx), ((generatedRoot.w_206_3 ctx), ((generatedRoot.w_206_4 ctx), ((generatedRoot.w_206_5 ctx), ())))))) (((generatedRoot.w_207_0 ctx) i), (((generatedRoot.w_207_1 ctx) i), (((generatedRoot.w_207_2 ctx) i), (((generatedRoot.w_207_3 ctx) i), (((generatedRoot.w_207_4 ctx) i), (((generatedRoot.w_207_5 ctx) i), ())))))))
  t235 : (2607022080) = 2607022080
  h210_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_166_0 ctx) i) ∧ ((generatedRoot.w_166_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_166_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_166_0 ctx) i) ∧ ((generatedRoot.w_166_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_166_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_166_0 ctx) i) ∧ ((generatedRoot.w_166_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_166_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_210 backend hashModel params i (((generatedRoot.w_166_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_207_0 ctx), ((generatedRoot.w_207_2 ctx), ((generatedRoot.w_207_4 ctx), (w_47_0_gathered, ((generatedRoot.w_207_1 ctx), ((generatedRoot.w_207_3 ctx), ((generatedRoot.w_207_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_208_0 ctx) i), (((generatedRoot.w_209_0 ctx) i), ())))))))))))))) (((generatedRoot.w_210_0 ctx) i), (((generatedRoot.w_210_1 ctx) i), (((generatedRoot.w_210_2 ctx) i), (((generatedRoot.w_210_3 ctx) i), (((generatedRoot.w_210_4 ctx) i), (((generatedRoot.w_210_5 ctx) i), ())))))))
  t237 : (20054016) = 20054016
  h211_5 : (∀ i : Fin 20054016, parallel_generatedRoot_211 backend hashModel params i ((generatedRoot.w_210_0 ctx), ((generatedRoot.w_210_1 ctx), ((generatedRoot.w_210_2 ctx), ((generatedRoot.w_210_3 ctx), ((generatedRoot.w_210_4 ctx), ((generatedRoot.w_210_5 ctx), ())))))) (((generatedRoot.w_211_0 ctx) i), (((generatedRoot.w_211_1 ctx) i), (((generatedRoot.w_211_2 ctx) i), (((generatedRoot.w_211_3 ctx) i), (((generatedRoot.w_211_4 ctx) i), (((generatedRoot.w_211_5 ctx) i), ())))))))
  t239 : (2607022080) = 2607022080
  h214_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_165_0 ctx) i) ∧ ((generatedRoot.w_165_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_165_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_165_0 ctx) i) ∧ ((generatedRoot.w_165_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_165_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_165_0 ctx) i) ∧ ((generatedRoot.w_165_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_165_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_214 backend hashModel params i (((generatedRoot.w_165_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_211_0 ctx), ((generatedRoot.w_211_2 ctx), ((generatedRoot.w_211_4 ctx), (w_47_0_gathered, ((generatedRoot.w_211_1 ctx), ((generatedRoot.w_211_3 ctx), ((generatedRoot.w_211_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_212_0 ctx) i), (((generatedRoot.w_213_0 ctx) i), ())))))))))))))) (((generatedRoot.w_214_0 ctx) i), (((generatedRoot.w_214_1 ctx) i), (((generatedRoot.w_214_2 ctx) i), (((generatedRoot.w_214_3 ctx) i), (((generatedRoot.w_214_4 ctx) i), (((generatedRoot.w_214_5 ctx) i), ())))))))
  t241 : (20054016) = 20054016
  h215_5 : (∀ i : Fin 20054016, parallel_generatedRoot_215 backend hashModel params i ((generatedRoot.w_214_0 ctx), ((generatedRoot.w_214_1 ctx), ((generatedRoot.w_214_2 ctx), ((generatedRoot.w_214_3 ctx), ((generatedRoot.w_214_4 ctx), ((generatedRoot.w_214_5 ctx), ())))))) (((generatedRoot.w_215_0 ctx) i), (((generatedRoot.w_215_1 ctx) i), (((generatedRoot.w_215_2 ctx) i), (((generatedRoot.w_215_3 ctx) i), (((generatedRoot.w_215_4 ctx) i), (((generatedRoot.w_215_5 ctx) i), ())))))))
  t243 : (2607022080) = 2607022080
  h218_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_164_0 ctx) i) ∧ ((generatedRoot.w_164_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_164_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_164_0 ctx) i) ∧ ((generatedRoot.w_164_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_164_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_164_0 ctx) i) ∧ ((generatedRoot.w_164_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_164_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_218 backend hashModel params i (((generatedRoot.w_164_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_215_0 ctx), ((generatedRoot.w_215_2 ctx), ((generatedRoot.w_215_4 ctx), (w_47_0_gathered, ((generatedRoot.w_215_1 ctx), ((generatedRoot.w_215_3 ctx), ((generatedRoot.w_215_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_216_0 ctx) i), (((generatedRoot.w_217_0 ctx) i), ())))))))))))))) (((generatedRoot.w_218_0 ctx) i), (((generatedRoot.w_218_1 ctx) i), (((generatedRoot.w_218_2 ctx) i), (((generatedRoot.w_218_3 ctx) i), (((generatedRoot.w_218_4 ctx) i), (((generatedRoot.w_218_5 ctx) i), ())))))))
  t245 : (20054016) = 20054016
  h219_5 : (∀ i : Fin 20054016, parallel_generatedRoot_219 backend hashModel params i ((generatedRoot.w_218_0 ctx), ((generatedRoot.w_218_1 ctx), ((generatedRoot.w_218_2 ctx), ((generatedRoot.w_218_3 ctx), ((generatedRoot.w_218_4 ctx), ((generatedRoot.w_218_5 ctx), ())))))) (((generatedRoot.w_219_0 ctx) i), (((generatedRoot.w_219_1 ctx) i), (((generatedRoot.w_219_2 ctx) i), (((generatedRoot.w_219_3 ctx) i), (((generatedRoot.w_219_4 ctx) i), (((generatedRoot.w_219_5 ctx) i), ())))))))
  t247 : (2607022080) = 2607022080
  h222_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_163_0 ctx) i) ∧ ((generatedRoot.w_163_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_163_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_163_0 ctx) i) ∧ ((generatedRoot.w_163_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_163_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_163_0 ctx) i) ∧ ((generatedRoot.w_163_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_163_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_222 backend hashModel params i (((generatedRoot.w_163_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_219_0 ctx), ((generatedRoot.w_219_2 ctx), ((generatedRoot.w_219_4 ctx), (w_47_0_gathered, ((generatedRoot.w_219_1 ctx), ((generatedRoot.w_219_3 ctx), ((generatedRoot.w_219_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_220_0 ctx) i), (((generatedRoot.w_221_0 ctx) i), ())))))))))))))) (((generatedRoot.w_222_0 ctx) i), (((generatedRoot.w_222_1 ctx) i), (((generatedRoot.w_222_2 ctx) i), (((generatedRoot.w_222_3 ctx) i), (((generatedRoot.w_222_4 ctx) i), (((generatedRoot.w_222_5 ctx) i), ())))))))
  t249 : (20054016) = 20054016
  h223_5 : (∀ i : Fin 20054016, parallel_generatedRoot_223 backend hashModel params i ((generatedRoot.w_222_0 ctx), ((generatedRoot.w_222_1 ctx), ((generatedRoot.w_222_2 ctx), ((generatedRoot.w_222_3 ctx), ((generatedRoot.w_222_4 ctx), ((generatedRoot.w_222_5 ctx), ())))))) (((generatedRoot.w_223_0 ctx) i), (((generatedRoot.w_223_1 ctx) i), (((generatedRoot.w_223_2 ctx) i), (((generatedRoot.w_223_3 ctx) i), (((generatedRoot.w_223_4 ctx) i), (((generatedRoot.w_223_5 ctx) i), ())))))))
  t251 : (2607022080) = 2607022080
  h226_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_162_0 ctx) i) ∧ ((generatedRoot.w_162_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_162_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_162_0 ctx) i) ∧ ((generatedRoot.w_162_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_162_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_162_0 ctx) i) ∧ ((generatedRoot.w_162_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_162_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_226 backend hashModel params i (((generatedRoot.w_162_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_223_0 ctx), ((generatedRoot.w_223_2 ctx), ((generatedRoot.w_223_4 ctx), (w_47_0_gathered, ((generatedRoot.w_223_1 ctx), ((generatedRoot.w_223_3 ctx), ((generatedRoot.w_223_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_224_0 ctx) i), (((generatedRoot.w_225_0 ctx) i), ())))))))))))))) (((generatedRoot.w_226_0 ctx) i), (((generatedRoot.w_226_1 ctx) i), (((generatedRoot.w_226_2 ctx) i), (((generatedRoot.w_226_3 ctx) i), (((generatedRoot.w_226_4 ctx) i), (((generatedRoot.w_226_5 ctx) i), ())))))))
  t253 : (20054016) = 20054016
  h227_5 : (∀ i : Fin 20054016, parallel_generatedRoot_227 backend hashModel params i ((generatedRoot.w_226_0 ctx), ((generatedRoot.w_226_1 ctx), ((generatedRoot.w_226_2 ctx), ((generatedRoot.w_226_3 ctx), ((generatedRoot.w_226_4 ctx), ((generatedRoot.w_226_5 ctx), ())))))) (((generatedRoot.w_227_0 ctx) i), (((generatedRoot.w_227_1 ctx) i), (((generatedRoot.w_227_2 ctx) i), (((generatedRoot.w_227_3 ctx) i), (((generatedRoot.w_227_4 ctx) i), (((generatedRoot.w_227_5 ctx) i), ())))))))
  t255 : (2607022080) = 2607022080

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_7 backend hashModel params (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_217_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_225_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_195_1 ctx) (generatedRoot.w_195_2 ctx) (generatedRoot.w_195_3 ctx) (generatedRoot.w_195_4 ctx) (generatedRoot.w_195_5 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_198_1 ctx) (generatedRoot.w_198_2 ctx) (generatedRoot.w_198_3 ctx) (generatedRoot.w_198_4 ctx) (generatedRoot.w_198_5 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_199_1 ctx) (generatedRoot.w_199_2 ctx) (generatedRoot.w_199_3 ctx) (generatedRoot.w_199_4 ctx) (generatedRoot.w_199_5 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_202_1 ctx) (generatedRoot.w_202_2 ctx) (generatedRoot.w_202_3 ctx) (generatedRoot.w_202_4 ctx) (generatedRoot.w_202_5 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_203_1 ctx) (generatedRoot.w_203_2 ctx) (generatedRoot.w_203_3 ctx) (generatedRoot.w_203_4 ctx) (generatedRoot.w_203_5 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_206_1 ctx) (generatedRoot.w_206_2 ctx) (generatedRoot.w_206_3 ctx) (generatedRoot.w_206_4 ctx) (generatedRoot.w_206_5 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_207_1 ctx) (generatedRoot.w_207_2 ctx) (generatedRoot.w_207_3 ctx) (generatedRoot.w_207_4 ctx) (generatedRoot.w_207_5 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_211_1 ctx) (generatedRoot.w_211_2 ctx) (generatedRoot.w_211_3 ctx) (generatedRoot.w_211_4 ctx) (generatedRoot.w_211_5 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_214_1 ctx) (generatedRoot.w_214_2 ctx) (generatedRoot.w_214_3 ctx) (generatedRoot.w_214_4 ctx) (generatedRoot.w_214_5 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_215_1 ctx) (generatedRoot.w_215_2 ctx) (generatedRoot.w_215_3 ctx) (generatedRoot.w_215_4 ctx) (generatedRoot.w_215_5 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_218_1 ctx) (generatedRoot.w_218_2 ctx) (generatedRoot.w_218_3 ctx) (generatedRoot.w_218_4 ctx) (generatedRoot.w_218_5 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_219_1 ctx) (generatedRoot.w_219_2 ctx) (generatedRoot.w_219_3 ctx) (generatedRoot.w_219_4 ctx) (generatedRoot.w_219_5 ctx) (generatedRoot.w_222_0 ctx) (generatedRoot.w_222_1 ctx) (generatedRoot.w_222_2 ctx) (generatedRoot.w_222_3 ctx) (generatedRoot.w_222_4 ctx) (generatedRoot.w_222_5 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_223_1 ctx) (generatedRoot.w_223_2 ctx) (generatedRoot.w_223_3 ctx) (generatedRoot.w_223_4 ctx) (generatedRoot.w_223_5 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_226_1 ctx) (generatedRoot.w_226_2 ctx) (generatedRoot.w_226_3 ctx) (generatedRoot.w_226_4 ctx) (generatedRoot.w_226_5 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_227_1 ctx) (generatedRoot.w_227_2 ctx) (generatedRoot.w_227_3 ctx) (generatedRoot.w_227_4 ctx) (generatedRoot.w_227_5 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8  (ctx : generatedRootCtx) : Prop where
  h230_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_161_0 ctx) i) ∧ ((generatedRoot.w_161_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_161_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_161_0 ctx) i) ∧ ((generatedRoot.w_161_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_161_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_161_0 ctx) i) ∧ ((generatedRoot.w_161_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_161_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_230 backend hashModel params i (((generatedRoot.w_161_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_227_0 ctx), ((generatedRoot.w_227_2 ctx), ((generatedRoot.w_227_4 ctx), (w_47_0_gathered, ((generatedRoot.w_227_1 ctx), ((generatedRoot.w_227_3 ctx), ((generatedRoot.w_227_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_228_0 ctx) i), (((generatedRoot.w_229_0 ctx) i), ())))))))))))))) (((generatedRoot.w_230_0 ctx) i), (((generatedRoot.w_230_1 ctx) i), (((generatedRoot.w_230_2 ctx) i), (((generatedRoot.w_230_3 ctx) i), (((generatedRoot.w_230_4 ctx) i), (((generatedRoot.w_230_5 ctx) i), ())))))))
  t257 : (20054016) = 20054016
  h231_5 : (∀ i : Fin 20054016, parallel_generatedRoot_231 backend hashModel params i ((generatedRoot.w_230_0 ctx), ((generatedRoot.w_230_1 ctx), ((generatedRoot.w_230_2 ctx), ((generatedRoot.w_230_3 ctx), ((generatedRoot.w_230_4 ctx), ((generatedRoot.w_230_5 ctx), ())))))) (((generatedRoot.w_231_0 ctx) i), (((generatedRoot.w_231_1 ctx) i), (((generatedRoot.w_231_2 ctx) i), (((generatedRoot.w_231_3 ctx) i), (((generatedRoot.w_231_4 ctx) i), (((generatedRoot.w_231_5 ctx) i), ())))))))
  t259 : (2607022080) = 2607022080
  h234_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_160_0 ctx) i) ∧ ((generatedRoot.w_160_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_160_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_160_0 ctx) i) ∧ ((generatedRoot.w_160_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_160_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_160_0 ctx) i) ∧ ((generatedRoot.w_160_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_160_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_234 backend hashModel params i (((generatedRoot.w_160_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_231_0 ctx), ((generatedRoot.w_231_2 ctx), ((generatedRoot.w_231_4 ctx), (w_47_0_gathered, ((generatedRoot.w_231_1 ctx), ((generatedRoot.w_231_3 ctx), ((generatedRoot.w_231_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_232_0 ctx) i), (((generatedRoot.w_233_0 ctx) i), ())))))))))))))) (((generatedRoot.w_234_0 ctx) i), (((generatedRoot.w_234_1 ctx) i), (((generatedRoot.w_234_2 ctx) i), (((generatedRoot.w_234_3 ctx) i), (((generatedRoot.w_234_4 ctx) i), (((generatedRoot.w_234_5 ctx) i), ())))))))
  t261 : (20054016) = 20054016
  h235_5 : (∀ i : Fin 20054016, parallel_generatedRoot_235 backend hashModel params i ((generatedRoot.w_234_0 ctx), ((generatedRoot.w_234_1 ctx), ((generatedRoot.w_234_2 ctx), ((generatedRoot.w_234_3 ctx), ((generatedRoot.w_234_4 ctx), ((generatedRoot.w_234_5 ctx), ())))))) (((generatedRoot.w_235_0 ctx) i), (((generatedRoot.w_235_1 ctx) i), (((generatedRoot.w_235_2 ctx) i), (((generatedRoot.w_235_3 ctx) i), (((generatedRoot.w_235_4 ctx) i), (((generatedRoot.w_235_5 ctx) i), ())))))))
  t263 : (2607022080) = 2607022080
  h238_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_159_0 ctx) i) ∧ ((generatedRoot.w_159_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_159_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_159_0 ctx) i) ∧ ((generatedRoot.w_159_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_159_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_159_0 ctx) i) ∧ ((generatedRoot.w_159_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_159_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_238 backend hashModel params i (((generatedRoot.w_159_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_235_0 ctx), ((generatedRoot.w_235_2 ctx), ((generatedRoot.w_235_4 ctx), (w_47_0_gathered, ((generatedRoot.w_235_1 ctx), ((generatedRoot.w_235_3 ctx), ((generatedRoot.w_235_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_236_0 ctx) i), (((generatedRoot.w_237_0 ctx) i), ())))))))))))))) (((generatedRoot.w_238_0 ctx) i), (((generatedRoot.w_238_1 ctx) i), (((generatedRoot.w_238_2 ctx) i), (((generatedRoot.w_238_3 ctx) i), (((generatedRoot.w_238_4 ctx) i), (((generatedRoot.w_238_5 ctx) i), ())))))))
  t265 : (20054016) = 20054016
  h239_5 : (∀ i : Fin 20054016, parallel_generatedRoot_239 backend hashModel params i ((generatedRoot.w_238_0 ctx), ((generatedRoot.w_238_1 ctx), ((generatedRoot.w_238_2 ctx), ((generatedRoot.w_238_3 ctx), ((generatedRoot.w_238_4 ctx), ((generatedRoot.w_238_5 ctx), ())))))) (((generatedRoot.w_239_0 ctx) i), (((generatedRoot.w_239_1 ctx) i), (((generatedRoot.w_239_2 ctx) i), (((generatedRoot.w_239_3 ctx) i), (((generatedRoot.w_239_4 ctx) i), (((generatedRoot.w_239_5 ctx) i), ())))))))
  t267 : (2607022080) = 2607022080
  h242_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_158_0 ctx) i) ∧ ((generatedRoot.w_158_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_158_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_158_0 ctx) i) ∧ ((generatedRoot.w_158_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_158_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_158_0 ctx) i) ∧ ((generatedRoot.w_158_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_158_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_242 backend hashModel params i (((generatedRoot.w_158_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_239_0 ctx), ((generatedRoot.w_239_2 ctx), ((generatedRoot.w_239_4 ctx), (w_47_0_gathered, ((generatedRoot.w_239_1 ctx), ((generatedRoot.w_239_3 ctx), ((generatedRoot.w_239_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_240_0 ctx) i), (((generatedRoot.w_241_0 ctx) i), ())))))))))))))) (((generatedRoot.w_242_0 ctx) i), (((generatedRoot.w_242_1 ctx) i), (((generatedRoot.w_242_2 ctx) i), (((generatedRoot.w_242_3 ctx) i), (((generatedRoot.w_242_4 ctx) i), (((generatedRoot.w_242_5 ctx) i), ())))))))
  t269 : (20054016) = 20054016
  h243_5 : (∀ i : Fin 20054016, parallel_generatedRoot_243 backend hashModel params i ((generatedRoot.w_242_0 ctx), ((generatedRoot.w_242_1 ctx), ((generatedRoot.w_242_2 ctx), ((generatedRoot.w_242_3 ctx), ((generatedRoot.w_242_4 ctx), ((generatedRoot.w_242_5 ctx), ())))))) (((generatedRoot.w_243_0 ctx) i), (((generatedRoot.w_243_1 ctx) i), (((generatedRoot.w_243_2 ctx) i), (((generatedRoot.w_243_3 ctx) i), (((generatedRoot.w_243_4 ctx) i), (((generatedRoot.w_243_5 ctx) i), ())))))))
  t271 : (2607022080) = 2607022080
  h246_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_157_0 ctx) i) ∧ ((generatedRoot.w_157_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_157_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_157_0 ctx) i) ∧ ((generatedRoot.w_157_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_157_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_157_0 ctx) i) ∧ ((generatedRoot.w_157_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_157_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_246 backend hashModel params i (((generatedRoot.w_157_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_243_0 ctx), ((generatedRoot.w_243_2 ctx), ((generatedRoot.w_243_4 ctx), (w_47_0_gathered, ((generatedRoot.w_243_1 ctx), ((generatedRoot.w_243_3 ctx), ((generatedRoot.w_243_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_244_0 ctx) i), (((generatedRoot.w_245_0 ctx) i), ())))))))))))))) (((generatedRoot.w_246_0 ctx) i), (((generatedRoot.w_246_1 ctx) i), (((generatedRoot.w_246_2 ctx) i), (((generatedRoot.w_246_3 ctx) i), (((generatedRoot.w_246_4 ctx) i), (((generatedRoot.w_246_5 ctx) i), ())))))))
  t273 : (20054016) = 20054016
  h247_5 : (∀ i : Fin 20054016, parallel_generatedRoot_247 backend hashModel params i ((generatedRoot.w_246_0 ctx), ((generatedRoot.w_246_1 ctx), ((generatedRoot.w_246_2 ctx), ((generatedRoot.w_246_3 ctx), ((generatedRoot.w_246_4 ctx), ((generatedRoot.w_246_5 ctx), ())))))) (((generatedRoot.w_247_0 ctx) i), (((generatedRoot.w_247_1 ctx) i), (((generatedRoot.w_247_2 ctx) i), (((generatedRoot.w_247_3 ctx) i), (((generatedRoot.w_247_4 ctx) i), (((generatedRoot.w_247_5 ctx) i), ())))))))
  t275 : (2607022080) = 2607022080
  h250_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_156_0 ctx) i) ∧ ((generatedRoot.w_156_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_156_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_156_0 ctx) i) ∧ ((generatedRoot.w_156_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_156_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_156_0 ctx) i) ∧ ((generatedRoot.w_156_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_156_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_250 backend hashModel params i (((generatedRoot.w_156_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_247_0 ctx), ((generatedRoot.w_247_2 ctx), ((generatedRoot.w_247_4 ctx), (w_47_0_gathered, ((generatedRoot.w_247_1 ctx), ((generatedRoot.w_247_3 ctx), ((generatedRoot.w_247_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_248_0 ctx) i), (((generatedRoot.w_249_0 ctx) i), ())))))))))))))) (((generatedRoot.w_250_0 ctx) i), (((generatedRoot.w_250_1 ctx) i), (((generatedRoot.w_250_2 ctx) i), (((generatedRoot.w_250_3 ctx) i), (((generatedRoot.w_250_4 ctx) i), (((generatedRoot.w_250_5 ctx) i), ())))))))
  t277 : (20054016) = 20054016
  h251_5 : (∀ i : Fin 20054016, parallel_generatedRoot_251 backend hashModel params i ((generatedRoot.w_250_0 ctx), ((generatedRoot.w_250_1 ctx), ((generatedRoot.w_250_2 ctx), ((generatedRoot.w_250_3 ctx), ((generatedRoot.w_250_4 ctx), ((generatedRoot.w_250_5 ctx), ())))))) (((generatedRoot.w_251_0 ctx) i), (((generatedRoot.w_251_1 ctx) i), (((generatedRoot.w_251_2 ctx) i), (((generatedRoot.w_251_3 ctx) i), (((generatedRoot.w_251_4 ctx) i), (((generatedRoot.w_251_5 ctx) i), ())))))))
  t279 : (2607022080) = 2607022080
  h254_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_155_0 ctx) i) ∧ ((generatedRoot.w_155_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_155_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_155_0 ctx) i) ∧ ((generatedRoot.w_155_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_155_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_155_0 ctx) i) ∧ ((generatedRoot.w_155_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_155_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_254 backend hashModel params i (((generatedRoot.w_155_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_251_0 ctx), ((generatedRoot.w_251_2 ctx), ((generatedRoot.w_251_4 ctx), (w_47_0_gathered, ((generatedRoot.w_251_1 ctx), ((generatedRoot.w_251_3 ctx), ((generatedRoot.w_251_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_252_0 ctx) i), (((generatedRoot.w_253_0 ctx) i), ())))))))))))))) (((generatedRoot.w_254_0 ctx) i), (((generatedRoot.w_254_1 ctx) i), (((generatedRoot.w_254_2 ctx) i), (((generatedRoot.w_254_3 ctx) i), (((generatedRoot.w_254_4 ctx) i), (((generatedRoot.w_254_5 ctx) i), ())))))))
  t281 : (20054016) = 20054016
  h255_5 : (∀ i : Fin 20054016, parallel_generatedRoot_255 backend hashModel params i ((generatedRoot.w_254_0 ctx), ((generatedRoot.w_254_1 ctx), ((generatedRoot.w_254_2 ctx), ((generatedRoot.w_254_3 ctx), ((generatedRoot.w_254_4 ctx), ((generatedRoot.w_254_5 ctx), ())))))) (((generatedRoot.w_255_0 ctx) i), (((generatedRoot.w_255_1 ctx) i), (((generatedRoot.w_255_2 ctx) i), (((generatedRoot.w_255_3 ctx) i), (((generatedRoot.w_255_4 ctx) i), (((generatedRoot.w_255_5 ctx) i), ())))))))
  t283 : (2607022080) = 2607022080
  h258_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_154_0 ctx) i) ∧ ((generatedRoot.w_154_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_154_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_154_0 ctx) i) ∧ ((generatedRoot.w_154_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_154_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_154_0 ctx) i) ∧ ((generatedRoot.w_154_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_154_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_258 backend hashModel params i (((generatedRoot.w_154_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_255_0 ctx), ((generatedRoot.w_255_2 ctx), ((generatedRoot.w_255_4 ctx), (w_47_0_gathered, ((generatedRoot.w_255_1 ctx), ((generatedRoot.w_255_3 ctx), ((generatedRoot.w_255_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_256_0 ctx) i), (((generatedRoot.w_257_0 ctx) i), ())))))))))))))) (((generatedRoot.w_258_0 ctx) i), (((generatedRoot.w_258_1 ctx) i), (((generatedRoot.w_258_2 ctx) i), (((generatedRoot.w_258_3 ctx) i), (((generatedRoot.w_258_4 ctx) i), (((generatedRoot.w_258_5 ctx) i), ())))))))
  t285 : (20054016) = 20054016
  h259_5 : (∀ i : Fin 20054016, parallel_generatedRoot_259 backend hashModel params i ((generatedRoot.w_258_0 ctx), ((generatedRoot.w_258_1 ctx), ((generatedRoot.w_258_2 ctx), ((generatedRoot.w_258_3 ctx), ((generatedRoot.w_258_4 ctx), ((generatedRoot.w_258_5 ctx), ())))))) (((generatedRoot.w_259_0 ctx) i), (((generatedRoot.w_259_1 ctx) i), (((generatedRoot.w_259_2 ctx) i), (((generatedRoot.w_259_3 ctx) i), (((generatedRoot.w_259_4 ctx) i), (((generatedRoot.w_259_5 ctx) i), ())))))))
  t287 : (2607022080) = 2607022080

theorem generatedRoot.facts_8  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_8 backend hashModel params (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_241_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_227_1 ctx) (generatedRoot.w_227_2 ctx) (generatedRoot.w_227_3 ctx) (generatedRoot.w_227_4 ctx) (generatedRoot.w_227_5 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_230_1 ctx) (generatedRoot.w_230_2 ctx) (generatedRoot.w_230_3 ctx) (generatedRoot.w_230_4 ctx) (generatedRoot.w_230_5 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_231_1 ctx) (generatedRoot.w_231_2 ctx) (generatedRoot.w_231_3 ctx) (generatedRoot.w_231_4 ctx) (generatedRoot.w_231_5 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_234_1 ctx) (generatedRoot.w_234_2 ctx) (generatedRoot.w_234_3 ctx) (generatedRoot.w_234_4 ctx) (generatedRoot.w_234_5 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_235_1 ctx) (generatedRoot.w_235_2 ctx) (generatedRoot.w_235_3 ctx) (generatedRoot.w_235_4 ctx) (generatedRoot.w_235_5 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_238_1 ctx) (generatedRoot.w_238_2 ctx) (generatedRoot.w_238_3 ctx) (generatedRoot.w_238_4 ctx) (generatedRoot.w_238_5 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_239_1 ctx) (generatedRoot.w_239_2 ctx) (generatedRoot.w_239_3 ctx) (generatedRoot.w_239_4 ctx) (generatedRoot.w_239_5 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_242_1 ctx) (generatedRoot.w_242_2 ctx) (generatedRoot.w_242_3 ctx) (generatedRoot.w_242_4 ctx) (generatedRoot.w_242_5 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_243_1 ctx) (generatedRoot.w_243_2 ctx) (generatedRoot.w_243_3 ctx) (generatedRoot.w_243_4 ctx) (generatedRoot.w_243_5 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_246_1 ctx) (generatedRoot.w_246_2 ctx) (generatedRoot.w_246_3 ctx) (generatedRoot.w_246_4 ctx) (generatedRoot.w_246_5 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_247_1 ctx) (generatedRoot.w_247_2 ctx) (generatedRoot.w_247_3 ctx) (generatedRoot.w_247_4 ctx) (generatedRoot.w_247_5 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_250_1 ctx) (generatedRoot.w_250_2 ctx) (generatedRoot.w_250_3 ctx) (generatedRoot.w_250_4 ctx) (generatedRoot.w_250_5 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_251_1 ctx) (generatedRoot.w_251_2 ctx) (generatedRoot.w_251_3 ctx) (generatedRoot.w_251_4 ctx) (generatedRoot.w_251_5 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_254_1 ctx) (generatedRoot.w_254_2 ctx) (generatedRoot.w_254_3 ctx) (generatedRoot.w_254_4 ctx) (generatedRoot.w_254_5 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_255_1 ctx) (generatedRoot.w_255_2 ctx) (generatedRoot.w_255_3 ctx) (generatedRoot.w_255_4 ctx) (generatedRoot.w_255_5 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_258_1 ctx) (generatedRoot.w_258_2 ctx) (generatedRoot.w_258_3 ctx) (generatedRoot.w_258_4 ctx) (generatedRoot.w_258_5 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_259_1 ctx) (generatedRoot.w_259_2 ctx) (generatedRoot.w_259_3 ctx) (generatedRoot.w_259_4 ctx) (generatedRoot.w_259_5 ctx)) :
    generatedRoot.Facts_8 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_9  (ctx : generatedRootCtx) : Prop where
  h262_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_153_0 ctx) i) ∧ ((generatedRoot.w_153_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_153_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_153_0 ctx) i) ∧ ((generatedRoot.w_153_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_153_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_153_0 ctx) i) ∧ ((generatedRoot.w_153_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_153_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_262 backend hashModel params i (((generatedRoot.w_153_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_259_0 ctx), ((generatedRoot.w_259_2 ctx), ((generatedRoot.w_259_4 ctx), (w_47_0_gathered, ((generatedRoot.w_259_1 ctx), ((generatedRoot.w_259_3 ctx), ((generatedRoot.w_259_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_260_0 ctx) i), (((generatedRoot.w_261_0 ctx) i), ())))))))))))))) (((generatedRoot.w_262_0 ctx) i), (((generatedRoot.w_262_1 ctx) i), (((generatedRoot.w_262_2 ctx) i), (((generatedRoot.w_262_3 ctx) i), (((generatedRoot.w_262_4 ctx) i), (((generatedRoot.w_262_5 ctx) i), ())))))))
  t289 : (20054016) = 20054016
  h263_5 : (∀ i : Fin 20054016, parallel_generatedRoot_263 backend hashModel params i ((generatedRoot.w_262_0 ctx), ((generatedRoot.w_262_1 ctx), ((generatedRoot.w_262_2 ctx), ((generatedRoot.w_262_3 ctx), ((generatedRoot.w_262_4 ctx), ((generatedRoot.w_262_5 ctx), ())))))) (((generatedRoot.w_263_0 ctx) i), (((generatedRoot.w_263_1 ctx) i), (((generatedRoot.w_263_2 ctx) i), (((generatedRoot.w_263_3 ctx) i), (((generatedRoot.w_263_4 ctx) i), (((generatedRoot.w_263_5 ctx) i), ())))))))
  t291 : (2607022080) = 2607022080
  h266_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_152_0 ctx) i) ∧ ((generatedRoot.w_152_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_152_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_152_0 ctx) i) ∧ ((generatedRoot.w_152_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_152_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_152_0 ctx) i) ∧ ((generatedRoot.w_152_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_152_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_266 backend hashModel params i (((generatedRoot.w_152_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_263_0 ctx), ((generatedRoot.w_263_2 ctx), ((generatedRoot.w_263_4 ctx), (w_47_0_gathered, ((generatedRoot.w_263_1 ctx), ((generatedRoot.w_263_3 ctx), ((generatedRoot.w_263_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_264_0 ctx) i), (((generatedRoot.w_265_0 ctx) i), ())))))))))))))) (((generatedRoot.w_266_0 ctx) i), (((generatedRoot.w_266_1 ctx) i), (((generatedRoot.w_266_2 ctx) i), (((generatedRoot.w_266_3 ctx) i), (((generatedRoot.w_266_4 ctx) i), (((generatedRoot.w_266_5 ctx) i), ())))))))
  t293 : (20054016) = 20054016
  h267_5 : (∀ i : Fin 20054016, parallel_generatedRoot_267 backend hashModel params i ((generatedRoot.w_266_0 ctx), ((generatedRoot.w_266_1 ctx), ((generatedRoot.w_266_2 ctx), ((generatedRoot.w_266_3 ctx), ((generatedRoot.w_266_4 ctx), ((generatedRoot.w_266_5 ctx), ())))))) (((generatedRoot.w_267_0 ctx) i), (((generatedRoot.w_267_1 ctx) i), (((generatedRoot.w_267_2 ctx) i), (((generatedRoot.w_267_3 ctx) i), (((generatedRoot.w_267_4 ctx) i), (((generatedRoot.w_267_5 ctx) i), ())))))))
  t295 : (2607022080) = 2607022080
  h270_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_151_0 ctx) i) ∧ ((generatedRoot.w_151_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_151_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_151_0 ctx) i) ∧ ((generatedRoot.w_151_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_151_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_151_0 ctx) i) ∧ ((generatedRoot.w_151_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_151_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_270 backend hashModel params i (((generatedRoot.w_151_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_267_0 ctx), ((generatedRoot.w_267_2 ctx), ((generatedRoot.w_267_4 ctx), (w_47_0_gathered, ((generatedRoot.w_267_1 ctx), ((generatedRoot.w_267_3 ctx), ((generatedRoot.w_267_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_268_0 ctx) i), (((generatedRoot.w_269_0 ctx) i), ())))))))))))))) (((generatedRoot.w_270_0 ctx) i), (((generatedRoot.w_270_1 ctx) i), (((generatedRoot.w_270_2 ctx) i), (((generatedRoot.w_270_3 ctx) i), (((generatedRoot.w_270_4 ctx) i), (((generatedRoot.w_270_5 ctx) i), ())))))))
  t297 : (20054016) = 20054016
  h271_5 : (∀ i : Fin 20054016, parallel_generatedRoot_271 backend hashModel params i ((generatedRoot.w_270_0 ctx), ((generatedRoot.w_270_1 ctx), ((generatedRoot.w_270_2 ctx), ((generatedRoot.w_270_3 ctx), ((generatedRoot.w_270_4 ctx), ((generatedRoot.w_270_5 ctx), ())))))) (((generatedRoot.w_271_0 ctx) i), (((generatedRoot.w_271_1 ctx) i), (((generatedRoot.w_271_2 ctx) i), (((generatedRoot.w_271_3 ctx) i), (((generatedRoot.w_271_4 ctx) i), (((generatedRoot.w_271_5 ctx) i), ())))))))
  t299 : (2607022080) = 2607022080
  h274_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_150_0 ctx) i) ∧ ((generatedRoot.w_150_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_150_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_150_0 ctx) i) ∧ ((generatedRoot.w_150_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_150_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_150_0 ctx) i) ∧ ((generatedRoot.w_150_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_150_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_274 backend hashModel params i (((generatedRoot.w_150_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_271_0 ctx), ((generatedRoot.w_271_2 ctx), ((generatedRoot.w_271_4 ctx), (w_47_0_gathered, ((generatedRoot.w_271_1 ctx), ((generatedRoot.w_271_3 ctx), ((generatedRoot.w_271_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_272_0 ctx) i), (((generatedRoot.w_273_0 ctx) i), ())))))))))))))) (((generatedRoot.w_274_0 ctx) i), (((generatedRoot.w_274_1 ctx) i), (((generatedRoot.w_274_2 ctx) i), (((generatedRoot.w_274_3 ctx) i), (((generatedRoot.w_274_4 ctx) i), (((generatedRoot.w_274_5 ctx) i), ())))))))
  t301 : (20054016) = 20054016
  h275_5 : (∀ i : Fin 20054016, parallel_generatedRoot_275 backend hashModel params i ((generatedRoot.w_274_0 ctx), ((generatedRoot.w_274_1 ctx), ((generatedRoot.w_274_2 ctx), ((generatedRoot.w_274_3 ctx), ((generatedRoot.w_274_4 ctx), ((generatedRoot.w_274_5 ctx), ())))))) (((generatedRoot.w_275_0 ctx) i), (((generatedRoot.w_275_1 ctx) i), (((generatedRoot.w_275_2 ctx) i), (((generatedRoot.w_275_3 ctx) i), (((generatedRoot.w_275_4 ctx) i), (((generatedRoot.w_275_5 ctx) i), ())))))))
  t303 : (2607022080) = 2607022080
  h278_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_149_0 ctx) i) ∧ ((generatedRoot.w_149_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_149_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_149_0 ctx) i) ∧ ((generatedRoot.w_149_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_149_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_149_0 ctx) i) ∧ ((generatedRoot.w_149_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_149_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_278 backend hashModel params i (((generatedRoot.w_149_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_275_0 ctx), ((generatedRoot.w_275_2 ctx), ((generatedRoot.w_275_4 ctx), (w_47_0_gathered, ((generatedRoot.w_275_1 ctx), ((generatedRoot.w_275_3 ctx), ((generatedRoot.w_275_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_276_0 ctx) i), (((generatedRoot.w_277_0 ctx) i), ())))))))))))))) (((generatedRoot.w_278_0 ctx) i), (((generatedRoot.w_278_1 ctx) i), (((generatedRoot.w_278_2 ctx) i), (((generatedRoot.w_278_3 ctx) i), (((generatedRoot.w_278_4 ctx) i), (((generatedRoot.w_278_5 ctx) i), ())))))))
  t305 : (20054016) = 20054016
  h279_5 : (∀ i : Fin 20054016, parallel_generatedRoot_279 backend hashModel params i ((generatedRoot.w_278_0 ctx), ((generatedRoot.w_278_1 ctx), ((generatedRoot.w_278_2 ctx), ((generatedRoot.w_278_3 ctx), ((generatedRoot.w_278_4 ctx), ((generatedRoot.w_278_5 ctx), ())))))) (((generatedRoot.w_279_0 ctx) i), (((generatedRoot.w_279_1 ctx) i), (((generatedRoot.w_279_2 ctx) i), (((generatedRoot.w_279_3 ctx) i), (((generatedRoot.w_279_4 ctx) i), (((generatedRoot.w_279_5 ctx) i), ())))))))
  t307 : (2607022080) = 2607022080
  h282_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_148_0 ctx) i) ∧ ((generatedRoot.w_148_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_148_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_148_0 ctx) i) ∧ ((generatedRoot.w_148_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_148_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_148_0 ctx) i) ∧ ((generatedRoot.w_148_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_148_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_282 backend hashModel params i (((generatedRoot.w_148_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_279_0 ctx), ((generatedRoot.w_279_2 ctx), ((generatedRoot.w_279_4 ctx), (w_47_0_gathered, ((generatedRoot.w_279_1 ctx), ((generatedRoot.w_279_3 ctx), ((generatedRoot.w_279_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_280_0 ctx) i), (((generatedRoot.w_281_0 ctx) i), ())))))))))))))) (((generatedRoot.w_282_0 ctx) i), (((generatedRoot.w_282_1 ctx) i), (((generatedRoot.w_282_2 ctx) i), (((generatedRoot.w_282_3 ctx) i), (((generatedRoot.w_282_4 ctx) i), (((generatedRoot.w_282_5 ctx) i), ())))))))
  t309 : (20054016) = 20054016
  h283_5 : (∀ i : Fin 20054016, parallel_generatedRoot_283 backend hashModel params i ((generatedRoot.w_282_0 ctx), ((generatedRoot.w_282_1 ctx), ((generatedRoot.w_282_2 ctx), ((generatedRoot.w_282_3 ctx), ((generatedRoot.w_282_4 ctx), ((generatedRoot.w_282_5 ctx), ())))))) (((generatedRoot.w_283_0 ctx) i), (((generatedRoot.w_283_1 ctx) i), (((generatedRoot.w_283_2 ctx) i), (((generatedRoot.w_283_3 ctx) i), (((generatedRoot.w_283_4 ctx) i), (((generatedRoot.w_283_5 ctx) i), ())))))))
  t311 : (2607022080) = 2607022080
  h286_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_147_0 ctx) i) ∧ ((generatedRoot.w_147_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_147_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_147_0 ctx) i) ∧ ((generatedRoot.w_147_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_147_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_147_0 ctx) i) ∧ ((generatedRoot.w_147_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_147_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_286 backend hashModel params i (((generatedRoot.w_147_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_283_0 ctx), ((generatedRoot.w_283_2 ctx), ((generatedRoot.w_283_4 ctx), (w_47_0_gathered, ((generatedRoot.w_283_1 ctx), ((generatedRoot.w_283_3 ctx), ((generatedRoot.w_283_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_284_0 ctx) i), (((generatedRoot.w_285_0 ctx) i), ())))))))))))))) (((generatedRoot.w_286_0 ctx) i), (((generatedRoot.w_286_1 ctx) i), (((generatedRoot.w_286_2 ctx) i), (((generatedRoot.w_286_3 ctx) i), (((generatedRoot.w_286_4 ctx) i), (((generatedRoot.w_286_5 ctx) i), ())))))))
  t313 : (20054016) = 20054016
  h287_5 : (∀ i : Fin 20054016, parallel_generatedRoot_287 backend hashModel params i ((generatedRoot.w_286_0 ctx), ((generatedRoot.w_286_1 ctx), ((generatedRoot.w_286_2 ctx), ((generatedRoot.w_286_3 ctx), ((generatedRoot.w_286_4 ctx), ((generatedRoot.w_286_5 ctx), ())))))) (((generatedRoot.w_287_0 ctx) i), (((generatedRoot.w_287_1 ctx) i), (((generatedRoot.w_287_2 ctx) i), (((generatedRoot.w_287_3 ctx) i), (((generatedRoot.w_287_4 ctx) i), (((generatedRoot.w_287_5 ctx) i), ())))))))
  t315 : (2607022080) = 2607022080
  h290_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_146_0 ctx) i) ∧ ((generatedRoot.w_146_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_146_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_146_0 ctx) i) ∧ ((generatedRoot.w_146_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_146_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_146_0 ctx) i) ∧ ((generatedRoot.w_146_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_146_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_290 backend hashModel params i (((generatedRoot.w_146_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_287_2 ctx), ((generatedRoot.w_287_4 ctx), (w_47_0_gathered, ((generatedRoot.w_287_1 ctx), ((generatedRoot.w_287_3 ctx), ((generatedRoot.w_287_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_288_0 ctx) i), (((generatedRoot.w_289_0 ctx) i), ())))))))))))))) (((generatedRoot.w_290_0 ctx) i), (((generatedRoot.w_290_1 ctx) i), (((generatedRoot.w_290_2 ctx) i), (((generatedRoot.w_290_3 ctx) i), (((generatedRoot.w_290_4 ctx) i), (((generatedRoot.w_290_5 ctx) i), ())))))))
  t317 : (20054016) = 20054016
  h291_5 : (∀ i : Fin 20054016, parallel_generatedRoot_291 backend hashModel params i ((generatedRoot.w_290_0 ctx), ((generatedRoot.w_290_1 ctx), ((generatedRoot.w_290_2 ctx), ((generatedRoot.w_290_3 ctx), ((generatedRoot.w_290_4 ctx), ((generatedRoot.w_290_5 ctx), ())))))) (((generatedRoot.w_291_0 ctx) i), (((generatedRoot.w_291_1 ctx) i), (((generatedRoot.w_291_2 ctx) i), (((generatedRoot.w_291_3 ctx) i), (((generatedRoot.w_291_4 ctx) i), (((generatedRoot.w_291_5 ctx) i), ())))))))
  t319 : (2607022080) = 2607022080

theorem generatedRoot.facts_9  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_9 backend hashModel params (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_273_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_281_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_289_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_259_1 ctx) (generatedRoot.w_259_2 ctx) (generatedRoot.w_259_3 ctx) (generatedRoot.w_259_4 ctx) (generatedRoot.w_259_5 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_262_1 ctx) (generatedRoot.w_262_2 ctx) (generatedRoot.w_262_3 ctx) (generatedRoot.w_262_4 ctx) (generatedRoot.w_262_5 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_263_1 ctx) (generatedRoot.w_263_2 ctx) (generatedRoot.w_263_3 ctx) (generatedRoot.w_263_4 ctx) (generatedRoot.w_263_5 ctx) (generatedRoot.w_266_0 ctx) (generatedRoot.w_266_1 ctx) (generatedRoot.w_266_2 ctx) (generatedRoot.w_266_3 ctx) (generatedRoot.w_266_4 ctx) (generatedRoot.w_266_5 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_267_1 ctx) (generatedRoot.w_267_2 ctx) (generatedRoot.w_267_3 ctx) (generatedRoot.w_267_4 ctx) (generatedRoot.w_267_5 ctx) (generatedRoot.w_270_0 ctx) (generatedRoot.w_270_1 ctx) (generatedRoot.w_270_2 ctx) (generatedRoot.w_270_3 ctx) (generatedRoot.w_270_4 ctx) (generatedRoot.w_270_5 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_271_1 ctx) (generatedRoot.w_271_2 ctx) (generatedRoot.w_271_3 ctx) (generatedRoot.w_271_4 ctx) (generatedRoot.w_271_5 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_274_1 ctx) (generatedRoot.w_274_2 ctx) (generatedRoot.w_274_3 ctx) (generatedRoot.w_274_4 ctx) (generatedRoot.w_274_5 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_275_1 ctx) (generatedRoot.w_275_2 ctx) (generatedRoot.w_275_3 ctx) (generatedRoot.w_275_4 ctx) (generatedRoot.w_275_5 ctx) (generatedRoot.w_278_0 ctx) (generatedRoot.w_278_1 ctx) (generatedRoot.w_278_2 ctx) (generatedRoot.w_278_3 ctx) (generatedRoot.w_278_4 ctx) (generatedRoot.w_278_5 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_279_1 ctx) (generatedRoot.w_279_2 ctx) (generatedRoot.w_279_3 ctx) (generatedRoot.w_279_4 ctx) (generatedRoot.w_279_5 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_282_1 ctx) (generatedRoot.w_282_2 ctx) (generatedRoot.w_282_3 ctx) (generatedRoot.w_282_4 ctx) (generatedRoot.w_282_5 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_283_1 ctx) (generatedRoot.w_283_2 ctx) (generatedRoot.w_283_3 ctx) (generatedRoot.w_283_4 ctx) (generatedRoot.w_283_5 ctx) (generatedRoot.w_286_0 ctx) (generatedRoot.w_286_1 ctx) (generatedRoot.w_286_2 ctx) (generatedRoot.w_286_3 ctx) (generatedRoot.w_286_4 ctx) (generatedRoot.w_286_5 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_287_1 ctx) (generatedRoot.w_287_2 ctx) (generatedRoot.w_287_3 ctx) (generatedRoot.w_287_4 ctx) (generatedRoot.w_287_5 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_290_1 ctx) (generatedRoot.w_290_2 ctx) (generatedRoot.w_290_3 ctx) (generatedRoot.w_290_4 ctx) (generatedRoot.w_290_5 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_291_1 ctx) (generatedRoot.w_291_2 ctx) (generatedRoot.w_291_3 ctx) (generatedRoot.w_291_4 ctx) (generatedRoot.w_291_5 ctx)) :
    generatedRoot.Facts_9 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_10  (ctx : generatedRootCtx) : Prop where
  h294_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_145_0 ctx) i) ∧ ((generatedRoot.w_145_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_145_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_145_0 ctx) i) ∧ ((generatedRoot.w_145_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_145_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_145_0 ctx) i) ∧ ((generatedRoot.w_145_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_145_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_294 backend hashModel params i (((generatedRoot.w_145_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_291_0 ctx), ((generatedRoot.w_291_2 ctx), ((generatedRoot.w_291_4 ctx), (w_47_0_gathered, ((generatedRoot.w_291_1 ctx), ((generatedRoot.w_291_3 ctx), ((generatedRoot.w_291_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_292_0 ctx) i), (((generatedRoot.w_293_0 ctx) i), ())))))))))))))) (((generatedRoot.w_294_0 ctx) i), (((generatedRoot.w_294_1 ctx) i), (((generatedRoot.w_294_2 ctx) i), (((generatedRoot.w_294_3 ctx) i), (((generatedRoot.w_294_4 ctx) i), (((generatedRoot.w_294_5 ctx) i), ())))))))
  t321 : (20054016) = 20054016
  h295_5 : (∀ i : Fin 20054016, parallel_generatedRoot_295 backend hashModel params i ((generatedRoot.w_294_0 ctx), ((generatedRoot.w_294_1 ctx), ((generatedRoot.w_294_2 ctx), ((generatedRoot.w_294_3 ctx), ((generatedRoot.w_294_4 ctx), ((generatedRoot.w_294_5 ctx), ())))))) (((generatedRoot.w_295_0 ctx) i), (((generatedRoot.w_295_1 ctx) i), (((generatedRoot.w_295_2 ctx) i), (((generatedRoot.w_295_3 ctx) i), (((generatedRoot.w_295_4 ctx) i), (((generatedRoot.w_295_5 ctx) i), ())))))))
  t323 : (2607022080) = 2607022080
  h298_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_144_0 ctx) i) ∧ ((generatedRoot.w_144_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_144_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_144_0 ctx) i) ∧ ((generatedRoot.w_144_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_144_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_144_0 ctx) i) ∧ ((generatedRoot.w_144_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_144_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_298 backend hashModel params i (((generatedRoot.w_144_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_295_0 ctx), ((generatedRoot.w_295_2 ctx), ((generatedRoot.w_295_4 ctx), (w_47_0_gathered, ((generatedRoot.w_295_1 ctx), ((generatedRoot.w_295_3 ctx), ((generatedRoot.w_295_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_296_0 ctx) i), (((generatedRoot.w_297_0 ctx) i), ())))))))))))))) (((generatedRoot.w_298_0 ctx) i), (((generatedRoot.w_298_1 ctx) i), (((generatedRoot.w_298_2 ctx) i), (((generatedRoot.w_298_3 ctx) i), (((generatedRoot.w_298_4 ctx) i), (((generatedRoot.w_298_5 ctx) i), ())))))))
  t325 : (20054016) = 20054016
  h299_5 : (∀ i : Fin 20054016, parallel_generatedRoot_299 backend hashModel params i ((generatedRoot.w_298_0 ctx), ((generatedRoot.w_298_1 ctx), ((generatedRoot.w_298_2 ctx), ((generatedRoot.w_298_3 ctx), ((generatedRoot.w_298_4 ctx), ((generatedRoot.w_298_5 ctx), ())))))) (((generatedRoot.w_299_0 ctx) i), (((generatedRoot.w_299_1 ctx) i), (((generatedRoot.w_299_2 ctx) i), (((generatedRoot.w_299_3 ctx) i), (((generatedRoot.w_299_4 ctx) i), (((generatedRoot.w_299_5 ctx) i), ())))))))
  t327 : (2607022080) = 2607022080
  h302_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_143_0 ctx) i) ∧ ((generatedRoot.w_143_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_143_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_143_0 ctx) i) ∧ ((generatedRoot.w_143_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_143_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_143_0 ctx) i) ∧ ((generatedRoot.w_143_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_143_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_302 backend hashModel params i (((generatedRoot.w_143_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_299_0 ctx), ((generatedRoot.w_299_2 ctx), ((generatedRoot.w_299_4 ctx), (w_47_0_gathered, ((generatedRoot.w_299_1 ctx), ((generatedRoot.w_299_3 ctx), ((generatedRoot.w_299_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_300_0 ctx) i), (((generatedRoot.w_301_0 ctx) i), ())))))))))))))) (((generatedRoot.w_302_0 ctx) i), (((generatedRoot.w_302_1 ctx) i), (((generatedRoot.w_302_2 ctx) i), (((generatedRoot.w_302_3 ctx) i), (((generatedRoot.w_302_4 ctx) i), (((generatedRoot.w_302_5 ctx) i), ())))))))
  t329 : (20054016) = 20054016
  h303_5 : (∀ i : Fin 20054016, parallel_generatedRoot_303 backend hashModel params i ((generatedRoot.w_302_0 ctx), ((generatedRoot.w_302_1 ctx), ((generatedRoot.w_302_2 ctx), ((generatedRoot.w_302_3 ctx), ((generatedRoot.w_302_4 ctx), ((generatedRoot.w_302_5 ctx), ())))))) (((generatedRoot.w_303_0 ctx) i), (((generatedRoot.w_303_1 ctx) i), (((generatedRoot.w_303_2 ctx) i), (((generatedRoot.w_303_3 ctx) i), (((generatedRoot.w_303_4 ctx) i), (((generatedRoot.w_303_5 ctx) i), ())))))))
  t331 : (2607022080) = 2607022080
  h306_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_142_0 ctx) i) ∧ ((generatedRoot.w_142_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_142_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_142_0 ctx) i) ∧ ((generatedRoot.w_142_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_142_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_142_0 ctx) i) ∧ ((generatedRoot.w_142_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_142_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_306 backend hashModel params i (((generatedRoot.w_142_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_303_0 ctx), ((generatedRoot.w_303_2 ctx), ((generatedRoot.w_303_4 ctx), (w_47_0_gathered, ((generatedRoot.w_303_1 ctx), ((generatedRoot.w_303_3 ctx), ((generatedRoot.w_303_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_304_0 ctx) i), (((generatedRoot.w_305_0 ctx) i), ())))))))))))))) (((generatedRoot.w_306_0 ctx) i), (((generatedRoot.w_306_1 ctx) i), (((generatedRoot.w_306_2 ctx) i), (((generatedRoot.w_306_3 ctx) i), (((generatedRoot.w_306_4 ctx) i), (((generatedRoot.w_306_5 ctx) i), ())))))))
  t333 : (20054016) = 20054016
  h307_5 : (∀ i : Fin 20054016, parallel_generatedRoot_307 backend hashModel params i ((generatedRoot.w_306_0 ctx), ((generatedRoot.w_306_1 ctx), ((generatedRoot.w_306_2 ctx), ((generatedRoot.w_306_3 ctx), ((generatedRoot.w_306_4 ctx), ((generatedRoot.w_306_5 ctx), ())))))) (((generatedRoot.w_307_0 ctx) i), (((generatedRoot.w_307_1 ctx) i), (((generatedRoot.w_307_2 ctx) i), (((generatedRoot.w_307_3 ctx) i), (((generatedRoot.w_307_4 ctx) i), (((generatedRoot.w_307_5 ctx) i), ())))))))
  t335 : (2607022080) = 2607022080
  h310_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_141_0 ctx) i) ∧ ((generatedRoot.w_141_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_141_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_141_0 ctx) i) ∧ ((generatedRoot.w_141_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_141_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_141_0 ctx) i) ∧ ((generatedRoot.w_141_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_141_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_310 backend hashModel params i (((generatedRoot.w_141_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_307_0 ctx), ((generatedRoot.w_307_2 ctx), ((generatedRoot.w_307_4 ctx), (w_47_0_gathered, ((generatedRoot.w_307_1 ctx), ((generatedRoot.w_307_3 ctx), ((generatedRoot.w_307_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_308_0 ctx) i), (((generatedRoot.w_309_0 ctx) i), ())))))))))))))) (((generatedRoot.w_310_0 ctx) i), (((generatedRoot.w_310_1 ctx) i), (((generatedRoot.w_310_2 ctx) i), (((generatedRoot.w_310_3 ctx) i), (((generatedRoot.w_310_4 ctx) i), (((generatedRoot.w_310_5 ctx) i), ())))))))
  t337 : (20054016) = 20054016
  h311_5 : (∀ i : Fin 20054016, parallel_generatedRoot_311 backend hashModel params i ((generatedRoot.w_310_0 ctx), ((generatedRoot.w_310_1 ctx), ((generatedRoot.w_310_2 ctx), ((generatedRoot.w_310_3 ctx), ((generatedRoot.w_310_4 ctx), ((generatedRoot.w_310_5 ctx), ())))))) (((generatedRoot.w_311_0 ctx) i), (((generatedRoot.w_311_1 ctx) i), (((generatedRoot.w_311_2 ctx) i), (((generatedRoot.w_311_3 ctx) i), (((generatedRoot.w_311_4 ctx) i), (((generatedRoot.w_311_5 ctx) i), ())))))))
  t339 : (2607022080) = 2607022080
  h314_5 : (∀ i : Fin 2607022080, ∃ w_16_0_gathered : mxxMat0, ∃ w_27_0_gathered : mxxMat0, ∃ w_47_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_140_0 ctx) i) ∧ ((generatedRoot.w_140_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_16_0 ctx) ((generatedRoot.w_140_0 ctx) i) w_16_0_gathered ∧ 0 ≤ ((generatedRoot.w_140_0 ctx) i) ∧ ((generatedRoot.w_140_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_27_0 ctx) ((generatedRoot.w_140_0 ctx) i) w_27_0_gathered ∧ 0 ≤ ((generatedRoot.w_140_0 ctx) i) ∧ ((generatedRoot.w_140_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_47_0 ctx) ((generatedRoot.w_140_0 ctx) i) w_47_0_gathered ∧ parallel_generatedRoot_314 backend hashModel params i (((generatedRoot.w_140_0 ctx) i), (w_16_0_gathered, ((generatedRoot.w_14_0 ctx), (w_27_0_gathered, ((generatedRoot.w_311_0 ctx), ((generatedRoot.w_311_2 ctx), ((generatedRoot.w_311_4 ctx), (w_47_0_gathered, ((generatedRoot.w_311_1 ctx), ((generatedRoot.w_311_3 ctx), ((generatedRoot.w_311_5 ctx), ((generatedRoot.w_70_0 ctx), (((generatedRoot.w_312_0 ctx) i), (((generatedRoot.w_313_0 ctx) i), ())))))))))))))) (((generatedRoot.w_314_0 ctx) i), (((generatedRoot.w_314_1 ctx) i), (((generatedRoot.w_314_2 ctx) i), (((generatedRoot.w_314_3 ctx) i), (((generatedRoot.w_314_4 ctx) i), (((generatedRoot.w_314_5 ctx) i), ())))))))
  t341 : (20054016) = 20054016
  h315_5 : (∀ i : Fin 20054016, parallel_generatedRoot_315 backend hashModel params i ((generatedRoot.w_314_0 ctx), ((generatedRoot.w_314_1 ctx), ((generatedRoot.w_314_2 ctx), ((generatedRoot.w_314_3 ctx), ((generatedRoot.w_314_4 ctx), ((generatedRoot.w_314_5 ctx), ())))))) (((generatedRoot.w_315_0 ctx) i), (((generatedRoot.w_315_1 ctx) i), (((generatedRoot.w_315_2 ctx) i), (((generatedRoot.w_315_3 ctx) i), (((generatedRoot.w_315_4 ctx) i), (((generatedRoot.w_315_5 ctx) i), ())))))))
  t343 : (20054016) = 20054016
  h316_5 : (∀ i : Fin 20054016, parallel_generatedRoot_316 backend hashModel params i ((generatedRoot.w_134_0 ctx), ((generatedRoot.w_135_0 ctx), ((generatedRoot.w_136_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_138_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_139_0 ctx), (((generatedRoot.w_315_4 ctx) i), (((generatedRoot.w_315_0 ctx) i), (((generatedRoot.w_315_2 ctx) i), (((generatedRoot.w_315_5 ctx) i), (((generatedRoot.w_315_1 ctx) i), (((generatedRoot.w_315_3 ctx) i), ()))))))))))))) (((generatedRoot.w_316_0 ctx) i), (((generatedRoot.w_316_1 ctx) i), (((generatedRoot.w_316_2 ctx) i), (((generatedRoot.w_316_3 ctx) i), (((generatedRoot.w_316_4 ctx) i), (((generatedRoot.w_316_5 ctx) i), ())))))))
  t345 : (20054016) = 20054016
  h317_5 : (∀ i : Fin 20054016, parallel_generatedRoot_317 backend hashModel params i ((generatedRoot.w_316_0 ctx), ((generatedRoot.w_316_1 ctx), ((generatedRoot.w_316_2 ctx), ((generatedRoot.w_316_3 ctx), ((generatedRoot.w_316_4 ctx), ((generatedRoot.w_316_5 ctx), ())))))) (((generatedRoot.w_317_0 ctx) i), (((generatedRoot.w_317_1 ctx) i), (((generatedRoot.w_317_2 ctx) i), (((generatedRoot.w_317_3 ctx) i), (((generatedRoot.w_317_4 ctx) i), (((generatedRoot.w_317_5 ctx) i), ())))))))
  h318_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_0 ctx) (0) (generatedRoot.w_318_0 ctx)
  h319_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_0 ctx) (1) (generatedRoot.w_319_0 ctx)
  h321_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_1 ctx) (0) (generatedRoot.w_321_0 ctx)
  h322_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_1 ctx) (1) (generatedRoot.w_322_0 ctx)
  h324_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_2 ctx) (0) (generatedRoot.w_324_0 ctx)

theorem generatedRoot.facts_10  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_10 backend hashModel params (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_291_1 ctx) (generatedRoot.w_291_2 ctx) (generatedRoot.w_291_3 ctx) (generatedRoot.w_291_4 ctx) (generatedRoot.w_291_5 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_294_1 ctx) (generatedRoot.w_294_2 ctx) (generatedRoot.w_294_3 ctx) (generatedRoot.w_294_4 ctx) (generatedRoot.w_294_5 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_295_1 ctx) (generatedRoot.w_295_2 ctx) (generatedRoot.w_295_3 ctx) (generatedRoot.w_295_4 ctx) (generatedRoot.w_295_5 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_298_1 ctx) (generatedRoot.w_298_2 ctx) (generatedRoot.w_298_3 ctx) (generatedRoot.w_298_4 ctx) (generatedRoot.w_298_5 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_299_1 ctx) (generatedRoot.w_299_2 ctx) (generatedRoot.w_299_3 ctx) (generatedRoot.w_299_4 ctx) (generatedRoot.w_299_5 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_302_1 ctx) (generatedRoot.w_302_2 ctx) (generatedRoot.w_302_3 ctx) (generatedRoot.w_302_4 ctx) (generatedRoot.w_302_5 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_303_1 ctx) (generatedRoot.w_303_2 ctx) (generatedRoot.w_303_3 ctx) (generatedRoot.w_303_4 ctx) (generatedRoot.w_303_5 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_306_1 ctx) (generatedRoot.w_306_2 ctx) (generatedRoot.w_306_3 ctx) (generatedRoot.w_306_4 ctx) (generatedRoot.w_306_5 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_307_1 ctx) (generatedRoot.w_307_2 ctx) (generatedRoot.w_307_3 ctx) (generatedRoot.w_307_4 ctx) (generatedRoot.w_307_5 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_310_1 ctx) (generatedRoot.w_310_2 ctx) (generatedRoot.w_310_3 ctx) (generatedRoot.w_310_4 ctx) (generatedRoot.w_310_5 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_311_1 ctx) (generatedRoot.w_311_2 ctx) (generatedRoot.w_311_3 ctx) (generatedRoot.w_311_4 ctx) (generatedRoot.w_311_5 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_314_1 ctx) (generatedRoot.w_314_2 ctx) (generatedRoot.w_314_3 ctx) (generatedRoot.w_314_4 ctx) (generatedRoot.w_314_5 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_315_1 ctx) (generatedRoot.w_315_2 ctx) (generatedRoot.w_315_3 ctx) (generatedRoot.w_315_4 ctx) (generatedRoot.w_315_5 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_316_1 ctx) (generatedRoot.w_316_2 ctx) (generatedRoot.w_316_3 ctx) (generatedRoot.w_316_4 ctx) (generatedRoot.w_316_5 ctx) (generatedRoot.w_317_0 ctx) (generatedRoot.w_317_1 ctx) (generatedRoot.w_317_2 ctx) (generatedRoot.w_317_3 ctx) (generatedRoot.w_317_4 ctx) (generatedRoot.w_317_5 ctx) (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_324_0 ctx)) :
    generatedRoot.Facts_10 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_11  (ctx : generatedRootCtx) : Prop where
  h325_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_2 ctx) (1) (generatedRoot.w_325_0 ctx)
  h327_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_3 ctx) (0) (generatedRoot.w_327_0 ctx)
  h328_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_3 ctx) (1) (generatedRoot.w_328_0 ctx)
  h330_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_4 ctx) (0) (generatedRoot.w_330_0 ctx)
  h331_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_4 ctx) (1) (generatedRoot.w_331_0 ctx)
  h333_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_5 ctx) (0) (generatedRoot.w_333_0 ctx)
  h334_0 : MxxRuntime.familyGetStatic (generatedRoot.w_317_5 ctx) (1) (generatedRoot.w_334_0 ctx)
  t359 : 0 ≤ (65534)
  h336_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_336 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_317_0 ctx), ((generatedRoot.w_317_1 ctx), ((generatedRoot.w_317_2 ctx), ((generatedRoot.w_317_3 ctx), ((generatedRoot.w_317_4 ctx), ((generatedRoot.w_317_5 ctx), ())))))))))))) next) (Int.toNat (65534)) ((generatedRoot.w_320_0 ctx), ((generatedRoot.w_323_0 ctx), ((generatedRoot.w_326_0 ctx), ((generatedRoot.w_329_0 ctx), ((generatedRoot.w_332_0 ctx), ((generatedRoot.w_335_0 ctx), ())))))) ((generatedRoot.w_336_0 ctx), ((generatedRoot.w_336_1 ctx), ((generatedRoot.w_336_2 ctx), ((generatedRoot.w_336_3 ctx), ((generatedRoot.w_336_4 ctx), ((generatedRoot.w_336_5 ctx), ()))))))
  h337_0 : MxxRuntime.tensorRuns (generatedRoot.w_133_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_337_0 ctx)
  t362 : 0 ≤ (0)
  t363 : (0) < 306
  h339_0 : MxxRuntime.tensorRuns (generatedRoot.w_338_0 ctx) (generatedRoot.w_336_1 ctx) (generatedRoot.w_339_0 ctx)
  t365 : 0 ≤ (0)
  t366 : (0) < 306
  h341_0 : MxxRuntime.tensorRuns (generatedRoot.w_340_0 ctx) (generatedRoot.w_336_2 ctx) (generatedRoot.w_341_0 ctx)
  t368 : 0 ≤ (0)
  t369 : (0) < 306
  h343_0 : MxxRuntime.tensorRuns (generatedRoot.w_342_0 ctx) (generatedRoot.w_336_3 ctx) (generatedRoot.w_343_0 ctx)
  t371 : 0 ≤ (0)
  t372 : (0) < 306
  h345_0 : MxxRuntime.tensorRuns (generatedRoot.w_344_0 ctx) (generatedRoot.w_336_4 ctx) (generatedRoot.w_345_0 ctx)
  t374 : 0 ≤ (0)
  t375 : (0) < 306
  h347_0 : MxxRuntime.tensorRuns (generatedRoot.w_346_0 ctx) (generatedRoot.w_336_5 ctx) (generatedRoot.w_347_0 ctx)
  t377 : 0 ≤ (305)
  h348_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit) => sequential_generatedRoot_348 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_317_0 ctx), ((generatedRoot.w_317_1 ctx), ((generatedRoot.w_317_2 ctx), ((generatedRoot.w_317_3 ctx), ((generatedRoot.w_317_4 ctx), ((generatedRoot.w_317_5 ctx), ())))))))))))) next) (Int.toNat (305)) ((generatedRoot.w_337_0 ctx), ((generatedRoot.w_339_0 ctx), ((generatedRoot.w_341_0 ctx), ((generatedRoot.w_343_0 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_347_0 ctx), ())))))) ((generatedRoot.w_348_0 ctx), ((generatedRoot.w_348_1 ctx), ((generatedRoot.w_348_2 ctx), ((generatedRoot.w_348_3 ctx), ((generatedRoot.w_348_4 ctx), ((generatedRoot.w_348_5 ctx), ()))))))
  t379 : (306) = 306
  h350_0 : (∀ i : Fin 306, parallel_generatedRoot_350 backend hashModel params i (generatedRoot.w_349_0 ctx) ((generatedRoot.w_350_0 ctx) i))
  t381 : 0 ≤ (306)
  h351_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat7 × mxxMat7 × mxxMat7 × Unit) => sequential_generatedRoot_351 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, ((generatedRoot.w_348_5 ctx), ((generatedRoot.w_350_0 ctx), ((generatedRoot.w_348_1 ctx), ((generatedRoot.w_348_3 ctx), ()))))))) next) (Int.toNat (306)) ((generatedRoot.w_132_0 ctx), ((generatedRoot.w_132_0 ctx), ((generatedRoot.w_132_0 ctx), ()))) ((generatedRoot.w_351_0 ctx), ((generatedRoot.w_351_1 ctx), ((generatedRoot.w_351_2 ctx), ())))
  h356_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx)

theorem generatedRoot.facts_11  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_11 backend hashModel params (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_317_0 ctx) (generatedRoot.w_317_1 ctx) (generatedRoot.w_317_2 ctx) (generatedRoot.w_317_3 ctx) (generatedRoot.w_317_4 ctx) (generatedRoot.w_317_5 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_325_0 ctx) (generatedRoot.w_326_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_329_0 ctx) (generatedRoot.w_330_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_333_0 ctx) (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_336_1 ctx) (generatedRoot.w_336_2 ctx) (generatedRoot.w_336_3 ctx) (generatedRoot.w_336_4 ctx) (generatedRoot.w_336_5 ctx) (generatedRoot.w_337_0 ctx) (generatedRoot.w_338_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_341_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_346_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_348_1 ctx) (generatedRoot.w_348_2 ctx) (generatedRoot.w_348_3 ctx) (generatedRoot.w_348_4 ctx) (generatedRoot.w_348_5 ctx) (generatedRoot.w_349_0 ctx) (generatedRoot.w_350_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_351_1 ctx) (generatedRoot.w_351_2 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx)) :
    generatedRoot.Facts_11 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_12  (ctx : generatedRootCtx) : Prop where
  h357_0 : MxxRuntime.concatRows (generatedRoot.w_91_6 ctx) (generatedRoot.w_91_8 ctx) (generatedRoot.w_357_0 ctx)
  h358_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_357_0 ctx) (generatedRoot.w_358_0 ctx)
  h359_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx)
  t387 : 0 ≤ 1
  t388 : 1 < 2
  t389 : 2 ≤ 3
  t390 : 0 ≤ 0
  t391 : 0 < 270
  t392 : 270 ≤ 270
  h361_0 : MxxRuntime.sliceMatrix (generatedRoot.w_360_0 ctx) 1 2 0 270 (generatedRoot.w_361_0 ctx)
  t394 : 0 ≤ 2
  t395 : 2 < 3
  t396 : 3 ≤ 3
  t397 : 0 ≤ 0
  t398 : 0 < 270
  t399 : 270 ≤ 270
  h362_0 : MxxRuntime.sliceMatrix (generatedRoot.w_360_0 ctx) 2 3 0 270 (generatedRoot.w_362_0 ctx)
  h363_0 : MxxRuntime.concatRows (generatedRoot.w_361_0 ctx) (generatedRoot.w_362_0 ctx) (generatedRoot.w_363_0 ctx)
  h364_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx)
  h365_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_364_0 ctx) (generatedRoot.w_365_0 ctx)
  t404 : 0 ≤ 1
  t405 : 1 < 2
  t406 : 2 ≤ 3
  t407 : 0 ≤ 0
  t408 : 0 < 270
  t409 : 270 ≤ 270
  h367_0 : MxxRuntime.sliceMatrix (generatedRoot.w_366_0 ctx) 1 2 0 270 (generatedRoot.w_367_0 ctx)
  t411 : 0 ≤ 1
  t412 : 1 < 2
  t413 : 2 ≤ 3
  t414 : 0 ≤ 0
  t415 : 0 < 270

theorem generatedRoot.facts_12  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_12 backend (generatedRoot.w_91_6 ctx) (generatedRoot.w_91_8 ctx) (generatedRoot.w_357_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_361_0 ctx) (generatedRoot.w_362_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_365_0 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx)) :
    generatedRoot.Facts_12 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_13  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t416 : 270 ≤ 270
  h369_0 : MxxRuntime.sliceMatrix (generatedRoot.w_120_0 ctx) 1 2 0 270 (generatedRoot.w_369_0 ctx)
  t418 : 0 ≤ 2
  t419 : 2 < 3
  t420 : 3 ≤ 3
  t421 : 0 ≤ 0
  t422 : 0 < 270
  t423 : 270 ≤ 270
  h371_0 : MxxRuntime.sliceMatrix (generatedRoot.w_366_0 ctx) 2 3 0 270 (generatedRoot.w_371_0 ctx)
  t425 : 0 ≤ 2
  t426 : 2 < 3
  t427 : 3 ≤ 3
  t428 : 0 ≤ 0
  t429 : 0 < 270
  t430 : 270 ≤ 270
  h373_0 : MxxRuntime.sliceMatrix (generatedRoot.w_120_0 ctx) 2 3 0 270 (generatedRoot.w_373_0 ctx)
  t432 : 0 ≤ 0
  t433 : 0 < 1
  t434 : 1 ≤ 3
  t435 : 0 ≤ 0
  t436 : 0 < 270
  t437 : 270 ≤ 270
  h375_0 : MxxRuntime.sliceMatrix (generatedRoot.w_366_0 ctx) 0 1 0 270 (generatedRoot.w_375_0 ctx)
  t439 : 0 ≤ 0
  t440 : 0 < 1
  t441 : 1 ≤ 3
  t442 : 0 ≤ 0
  t443 : 0 < 270
  t444 : 270 ≤ 270
  h376_0 : MxxRuntime.sliceMatrix (generatedRoot.w_360_0 ctx) 0 1 0 270 (generatedRoot.w_376_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_13  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_13 (generatedRoot.w_120_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_348_1 ctx) (generatedRoot.w_348_2 ctx) (generatedRoot.w_348_3 ctx) (generatedRoot.w_348_4 ctx) (generatedRoot.w_348_5 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_369_0 ctx) (generatedRoot.w_370_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_373_0 ctx) (generatedRoot.w_374_0 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_378_0 ctx) outputs) :
    generatedRoot.Facts_13 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

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
  c13 : generatedRoot.Facts_13 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body backend hashModel params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  ⟨(generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h.1.1.1), (generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) h.1.1.2.1), (generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) h.1.1.2.2), (generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) h.1.2.1.1), (generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) h.1.2.1.2), (generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) h.1.2.2.1), (generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) h.1.2.2.2), (generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) h.2.1.1), (generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) h.2.1.2.1), (generatedRoot.facts_9 (ctx := ctx) (outputs := outputs) h.2.1.2.2), (generatedRoot.facts_10 (ctx := ctx) (outputs := outputs) h.2.2.1.1), (generatedRoot.facts_11 (ctx := ctx) (outputs := outputs) h.2.2.1.2), (generatedRoot.facts_12 (ctx := ctx) (outputs := outputs) h.2.2.2.1), (generatedRoot.facts_13 (ctx := ctx) (outputs := outputs) h.2.2.2.2)⟩

theorem generatedRoot.h183_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_183.Witness, parallel_generatedRoot_183.Facts (⟨((generatedRoot.w_182_0 ctx), ((generatedRoot.w_182_1 ctx), ((generatedRoot.w_182_2 ctx), ((generatedRoot.w_182_3 ctx), ((generatedRoot.w_182_4 ctx), ((generatedRoot.w_182_5 ctx), ())))))), w⟩ : parallel_generatedRoot_183.Ctx backend hashModel params i) (((generatedRoot.w_183_0 ctx) i), (((generatedRoot.w_183_1 ctx) i), (((generatedRoot.w_183_2 ctx) i), (((generatedRoot.w_183_3 ctx) i), (((generatedRoot.w_183_4 ctx) i), (((generatedRoot.w_183_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h183_5 i
  ⟨w, parallel_generatedRoot_183.facts _ h⟩

theorem generatedRoot.h187_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_187.Witness, parallel_generatedRoot_187.Facts (⟨((generatedRoot.w_186_0 ctx), ((generatedRoot.w_186_1 ctx), ((generatedRoot.w_186_2 ctx), ((generatedRoot.w_186_3 ctx), ((generatedRoot.w_186_4 ctx), ((generatedRoot.w_186_5 ctx), ())))))), w⟩ : parallel_generatedRoot_187.Ctx backend hashModel params i) (((generatedRoot.w_187_0 ctx) i), (((generatedRoot.w_187_1 ctx) i), (((generatedRoot.w_187_2 ctx) i), (((generatedRoot.w_187_3 ctx) i), (((generatedRoot.w_187_4 ctx) i), (((generatedRoot.w_187_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h187_5 i
  ⟨w, parallel_generatedRoot_187.facts _ h⟩

theorem generatedRoot.h191_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_191.Witness, parallel_generatedRoot_191.Facts (⟨((generatedRoot.w_190_0 ctx), ((generatedRoot.w_190_1 ctx), ((generatedRoot.w_190_2 ctx), ((generatedRoot.w_190_3 ctx), ((generatedRoot.w_190_4 ctx), ((generatedRoot.w_190_5 ctx), ())))))), w⟩ : parallel_generatedRoot_191.Ctx backend hashModel params i) (((generatedRoot.w_191_0 ctx) i), (((generatedRoot.w_191_1 ctx) i), (((generatedRoot.w_191_2 ctx) i), (((generatedRoot.w_191_3 ctx) i), (((generatedRoot.w_191_4 ctx) i), (((generatedRoot.w_191_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h191_5 i
  ⟨w, parallel_generatedRoot_191.facts _ h⟩

theorem generatedRoot.h195_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_195.Witness, parallel_generatedRoot_195.Facts (⟨((generatedRoot.w_194_0 ctx), ((generatedRoot.w_194_1 ctx), ((generatedRoot.w_194_2 ctx), ((generatedRoot.w_194_3 ctx), ((generatedRoot.w_194_4 ctx), ((generatedRoot.w_194_5 ctx), ())))))), w⟩ : parallel_generatedRoot_195.Ctx backend hashModel params i) (((generatedRoot.w_195_0 ctx) i), (((generatedRoot.w_195_1 ctx) i), (((generatedRoot.w_195_2 ctx) i), (((generatedRoot.w_195_3 ctx) i), (((generatedRoot.w_195_4 ctx) i), (((generatedRoot.w_195_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h195_5 i
  ⟨w, parallel_generatedRoot_195.facts _ h⟩

theorem generatedRoot.h199_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_199.Witness, parallel_generatedRoot_199.Facts (⟨((generatedRoot.w_198_0 ctx), ((generatedRoot.w_198_1 ctx), ((generatedRoot.w_198_2 ctx), ((generatedRoot.w_198_3 ctx), ((generatedRoot.w_198_4 ctx), ((generatedRoot.w_198_5 ctx), ())))))), w⟩ : parallel_generatedRoot_199.Ctx backend hashModel params i) (((generatedRoot.w_199_0 ctx) i), (((generatedRoot.w_199_1 ctx) i), (((generatedRoot.w_199_2 ctx) i), (((generatedRoot.w_199_3 ctx) i), (((generatedRoot.w_199_4 ctx) i), (((generatedRoot.w_199_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h199_5 i
  ⟨w, parallel_generatedRoot_199.facts _ h⟩

theorem generatedRoot.h203_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_203.Witness, parallel_generatedRoot_203.Facts (⟨((generatedRoot.w_202_0 ctx), ((generatedRoot.w_202_1 ctx), ((generatedRoot.w_202_2 ctx), ((generatedRoot.w_202_3 ctx), ((generatedRoot.w_202_4 ctx), ((generatedRoot.w_202_5 ctx), ())))))), w⟩ : parallel_generatedRoot_203.Ctx backend hashModel params i) (((generatedRoot.w_203_0 ctx) i), (((generatedRoot.w_203_1 ctx) i), (((generatedRoot.w_203_2 ctx) i), (((generatedRoot.w_203_3 ctx) i), (((generatedRoot.w_203_4 ctx) i), (((generatedRoot.w_203_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h203_5 i
  ⟨w, parallel_generatedRoot_203.facts _ h⟩

theorem generatedRoot.h207_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_207.Witness, parallel_generatedRoot_207.Facts (⟨((generatedRoot.w_206_0 ctx), ((generatedRoot.w_206_1 ctx), ((generatedRoot.w_206_2 ctx), ((generatedRoot.w_206_3 ctx), ((generatedRoot.w_206_4 ctx), ((generatedRoot.w_206_5 ctx), ())))))), w⟩ : parallel_generatedRoot_207.Ctx backend hashModel params i) (((generatedRoot.w_207_0 ctx) i), (((generatedRoot.w_207_1 ctx) i), (((generatedRoot.w_207_2 ctx) i), (((generatedRoot.w_207_3 ctx) i), (((generatedRoot.w_207_4 ctx) i), (((generatedRoot.w_207_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h207_5 i
  ⟨w, parallel_generatedRoot_207.facts _ h⟩

theorem generatedRoot.h211_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_211.Witness, parallel_generatedRoot_211.Facts (⟨((generatedRoot.w_210_0 ctx), ((generatedRoot.w_210_1 ctx), ((generatedRoot.w_210_2 ctx), ((generatedRoot.w_210_3 ctx), ((generatedRoot.w_210_4 ctx), ((generatedRoot.w_210_5 ctx), ())))))), w⟩ : parallel_generatedRoot_211.Ctx backend hashModel params i) (((generatedRoot.w_211_0 ctx) i), (((generatedRoot.w_211_1 ctx) i), (((generatedRoot.w_211_2 ctx) i), (((generatedRoot.w_211_3 ctx) i), (((generatedRoot.w_211_4 ctx) i), (((generatedRoot.w_211_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h211_5 i
  ⟨w, parallel_generatedRoot_211.facts _ h⟩

theorem generatedRoot.h215_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_215.Witness, parallel_generatedRoot_215.Facts (⟨((generatedRoot.w_214_0 ctx), ((generatedRoot.w_214_1 ctx), ((generatedRoot.w_214_2 ctx), ((generatedRoot.w_214_3 ctx), ((generatedRoot.w_214_4 ctx), ((generatedRoot.w_214_5 ctx), ())))))), w⟩ : parallel_generatedRoot_215.Ctx backend hashModel params i) (((generatedRoot.w_215_0 ctx) i), (((generatedRoot.w_215_1 ctx) i), (((generatedRoot.w_215_2 ctx) i), (((generatedRoot.w_215_3 ctx) i), (((generatedRoot.w_215_4 ctx) i), (((generatedRoot.w_215_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h215_5 i
  ⟨w, parallel_generatedRoot_215.facts _ h⟩

theorem generatedRoot.h219_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_219.Witness, parallel_generatedRoot_219.Facts (⟨((generatedRoot.w_218_0 ctx), ((generatedRoot.w_218_1 ctx), ((generatedRoot.w_218_2 ctx), ((generatedRoot.w_218_3 ctx), ((generatedRoot.w_218_4 ctx), ((generatedRoot.w_218_5 ctx), ())))))), w⟩ : parallel_generatedRoot_219.Ctx backend hashModel params i) (((generatedRoot.w_219_0 ctx) i), (((generatedRoot.w_219_1 ctx) i), (((generatedRoot.w_219_2 ctx) i), (((generatedRoot.w_219_3 ctx) i), (((generatedRoot.w_219_4 ctx) i), (((generatedRoot.w_219_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h219_5 i
  ⟨w, parallel_generatedRoot_219.facts _ h⟩

theorem generatedRoot.h223_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_223.Witness, parallel_generatedRoot_223.Facts (⟨((generatedRoot.w_222_0 ctx), ((generatedRoot.w_222_1 ctx), ((generatedRoot.w_222_2 ctx), ((generatedRoot.w_222_3 ctx), ((generatedRoot.w_222_4 ctx), ((generatedRoot.w_222_5 ctx), ())))))), w⟩ : parallel_generatedRoot_223.Ctx backend hashModel params i) (((generatedRoot.w_223_0 ctx) i), (((generatedRoot.w_223_1 ctx) i), (((generatedRoot.w_223_2 ctx) i), (((generatedRoot.w_223_3 ctx) i), (((generatedRoot.w_223_4 ctx) i), (((generatedRoot.w_223_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h223_5 i
  ⟨w, parallel_generatedRoot_223.facts _ h⟩

theorem generatedRoot.h227_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_227.Witness, parallel_generatedRoot_227.Facts (⟨((generatedRoot.w_226_0 ctx), ((generatedRoot.w_226_1 ctx), ((generatedRoot.w_226_2 ctx), ((generatedRoot.w_226_3 ctx), ((generatedRoot.w_226_4 ctx), ((generatedRoot.w_226_5 ctx), ())))))), w⟩ : parallel_generatedRoot_227.Ctx backend hashModel params i) (((generatedRoot.w_227_0 ctx) i), (((generatedRoot.w_227_1 ctx) i), (((generatedRoot.w_227_2 ctx) i), (((generatedRoot.w_227_3 ctx) i), (((generatedRoot.w_227_4 ctx) i), (((generatedRoot.w_227_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h227_5 i
  ⟨w, parallel_generatedRoot_227.facts _ h⟩

theorem generatedRoot.h231_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_231.Witness, parallel_generatedRoot_231.Facts (⟨((generatedRoot.w_230_0 ctx), ((generatedRoot.w_230_1 ctx), ((generatedRoot.w_230_2 ctx), ((generatedRoot.w_230_3 ctx), ((generatedRoot.w_230_4 ctx), ((generatedRoot.w_230_5 ctx), ())))))), w⟩ : parallel_generatedRoot_231.Ctx backend hashModel params i) (((generatedRoot.w_231_0 ctx) i), (((generatedRoot.w_231_1 ctx) i), (((generatedRoot.w_231_2 ctx) i), (((generatedRoot.w_231_3 ctx) i), (((generatedRoot.w_231_4 ctx) i), (((generatedRoot.w_231_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h231_5 i
  ⟨w, parallel_generatedRoot_231.facts _ h⟩

theorem generatedRoot.h235_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_235.Witness, parallel_generatedRoot_235.Facts (⟨((generatedRoot.w_234_0 ctx), ((generatedRoot.w_234_1 ctx), ((generatedRoot.w_234_2 ctx), ((generatedRoot.w_234_3 ctx), ((generatedRoot.w_234_4 ctx), ((generatedRoot.w_234_5 ctx), ())))))), w⟩ : parallel_generatedRoot_235.Ctx backend hashModel params i) (((generatedRoot.w_235_0 ctx) i), (((generatedRoot.w_235_1 ctx) i), (((generatedRoot.w_235_2 ctx) i), (((generatedRoot.w_235_3 ctx) i), (((generatedRoot.w_235_4 ctx) i), (((generatedRoot.w_235_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h235_5 i
  ⟨w, parallel_generatedRoot_235.facts _ h⟩

theorem generatedRoot.h239_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_239.Witness, parallel_generatedRoot_239.Facts (⟨((generatedRoot.w_238_0 ctx), ((generatedRoot.w_238_1 ctx), ((generatedRoot.w_238_2 ctx), ((generatedRoot.w_238_3 ctx), ((generatedRoot.w_238_4 ctx), ((generatedRoot.w_238_5 ctx), ())))))), w⟩ : parallel_generatedRoot_239.Ctx backend hashModel params i) (((generatedRoot.w_239_0 ctx) i), (((generatedRoot.w_239_1 ctx) i), (((generatedRoot.w_239_2 ctx) i), (((generatedRoot.w_239_3 ctx) i), (((generatedRoot.w_239_4 ctx) i), (((generatedRoot.w_239_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h239_5 i
  ⟨w, parallel_generatedRoot_239.facts _ h⟩

theorem generatedRoot.h243_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_243.Witness, parallel_generatedRoot_243.Facts (⟨((generatedRoot.w_242_0 ctx), ((generatedRoot.w_242_1 ctx), ((generatedRoot.w_242_2 ctx), ((generatedRoot.w_242_3 ctx), ((generatedRoot.w_242_4 ctx), ((generatedRoot.w_242_5 ctx), ())))))), w⟩ : parallel_generatedRoot_243.Ctx backend hashModel params i) (((generatedRoot.w_243_0 ctx) i), (((generatedRoot.w_243_1 ctx) i), (((generatedRoot.w_243_2 ctx) i), (((generatedRoot.w_243_3 ctx) i), (((generatedRoot.w_243_4 ctx) i), (((generatedRoot.w_243_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h243_5 i
  ⟨w, parallel_generatedRoot_243.facts _ h⟩

theorem generatedRoot.h247_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_247.Witness, parallel_generatedRoot_247.Facts (⟨((generatedRoot.w_246_0 ctx), ((generatedRoot.w_246_1 ctx), ((generatedRoot.w_246_2 ctx), ((generatedRoot.w_246_3 ctx), ((generatedRoot.w_246_4 ctx), ((generatedRoot.w_246_5 ctx), ())))))), w⟩ : parallel_generatedRoot_247.Ctx backend hashModel params i) (((generatedRoot.w_247_0 ctx) i), (((generatedRoot.w_247_1 ctx) i), (((generatedRoot.w_247_2 ctx) i), (((generatedRoot.w_247_3 ctx) i), (((generatedRoot.w_247_4 ctx) i), (((generatedRoot.w_247_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h247_5 i
  ⟨w, parallel_generatedRoot_247.facts _ h⟩

theorem generatedRoot.h251_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_251.Witness, parallel_generatedRoot_251.Facts (⟨((generatedRoot.w_250_0 ctx), ((generatedRoot.w_250_1 ctx), ((generatedRoot.w_250_2 ctx), ((generatedRoot.w_250_3 ctx), ((generatedRoot.w_250_4 ctx), ((generatedRoot.w_250_5 ctx), ())))))), w⟩ : parallel_generatedRoot_251.Ctx backend hashModel params i) (((generatedRoot.w_251_0 ctx) i), (((generatedRoot.w_251_1 ctx) i), (((generatedRoot.w_251_2 ctx) i), (((generatedRoot.w_251_3 ctx) i), (((generatedRoot.w_251_4 ctx) i), (((generatedRoot.w_251_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h251_5 i
  ⟨w, parallel_generatedRoot_251.facts _ h⟩

theorem generatedRoot.h255_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_255.Witness, parallel_generatedRoot_255.Facts (⟨((generatedRoot.w_254_0 ctx), ((generatedRoot.w_254_1 ctx), ((generatedRoot.w_254_2 ctx), ((generatedRoot.w_254_3 ctx), ((generatedRoot.w_254_4 ctx), ((generatedRoot.w_254_5 ctx), ())))))), w⟩ : parallel_generatedRoot_255.Ctx backend hashModel params i) (((generatedRoot.w_255_0 ctx) i), (((generatedRoot.w_255_1 ctx) i), (((generatedRoot.w_255_2 ctx) i), (((generatedRoot.w_255_3 ctx) i), (((generatedRoot.w_255_4 ctx) i), (((generatedRoot.w_255_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h255_5 i
  ⟨w, parallel_generatedRoot_255.facts _ h⟩

theorem generatedRoot.h259_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_259.Witness, parallel_generatedRoot_259.Facts (⟨((generatedRoot.w_258_0 ctx), ((generatedRoot.w_258_1 ctx), ((generatedRoot.w_258_2 ctx), ((generatedRoot.w_258_3 ctx), ((generatedRoot.w_258_4 ctx), ((generatedRoot.w_258_5 ctx), ())))))), w⟩ : parallel_generatedRoot_259.Ctx backend hashModel params i) (((generatedRoot.w_259_0 ctx) i), (((generatedRoot.w_259_1 ctx) i), (((generatedRoot.w_259_2 ctx) i), (((generatedRoot.w_259_3 ctx) i), (((generatedRoot.w_259_4 ctx) i), (((generatedRoot.w_259_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h259_5 i
  ⟨w, parallel_generatedRoot_259.facts _ h⟩

theorem generatedRoot.h263_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_263.Witness, parallel_generatedRoot_263.Facts (⟨((generatedRoot.w_262_0 ctx), ((generatedRoot.w_262_1 ctx), ((generatedRoot.w_262_2 ctx), ((generatedRoot.w_262_3 ctx), ((generatedRoot.w_262_4 ctx), ((generatedRoot.w_262_5 ctx), ())))))), w⟩ : parallel_generatedRoot_263.Ctx backend hashModel params i) (((generatedRoot.w_263_0 ctx) i), (((generatedRoot.w_263_1 ctx) i), (((generatedRoot.w_263_2 ctx) i), (((generatedRoot.w_263_3 ctx) i), (((generatedRoot.w_263_4 ctx) i), (((generatedRoot.w_263_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h263_5 i
  ⟨w, parallel_generatedRoot_263.facts _ h⟩

theorem generatedRoot.h267_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_267.Witness, parallel_generatedRoot_267.Facts (⟨((generatedRoot.w_266_0 ctx), ((generatedRoot.w_266_1 ctx), ((generatedRoot.w_266_2 ctx), ((generatedRoot.w_266_3 ctx), ((generatedRoot.w_266_4 ctx), ((generatedRoot.w_266_5 ctx), ())))))), w⟩ : parallel_generatedRoot_267.Ctx backend hashModel params i) (((generatedRoot.w_267_0 ctx) i), (((generatedRoot.w_267_1 ctx) i), (((generatedRoot.w_267_2 ctx) i), (((generatedRoot.w_267_3 ctx) i), (((generatedRoot.w_267_4 ctx) i), (((generatedRoot.w_267_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h267_5 i
  ⟨w, parallel_generatedRoot_267.facts _ h⟩

theorem generatedRoot.h271_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_271.Witness, parallel_generatedRoot_271.Facts (⟨((generatedRoot.w_270_0 ctx), ((generatedRoot.w_270_1 ctx), ((generatedRoot.w_270_2 ctx), ((generatedRoot.w_270_3 ctx), ((generatedRoot.w_270_4 ctx), ((generatedRoot.w_270_5 ctx), ())))))), w⟩ : parallel_generatedRoot_271.Ctx backend hashModel params i) (((generatedRoot.w_271_0 ctx) i), (((generatedRoot.w_271_1 ctx) i), (((generatedRoot.w_271_2 ctx) i), (((generatedRoot.w_271_3 ctx) i), (((generatedRoot.w_271_4 ctx) i), (((generatedRoot.w_271_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h271_5 i
  ⟨w, parallel_generatedRoot_271.facts _ h⟩

theorem generatedRoot.h275_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_275.Witness, parallel_generatedRoot_275.Facts (⟨((generatedRoot.w_274_0 ctx), ((generatedRoot.w_274_1 ctx), ((generatedRoot.w_274_2 ctx), ((generatedRoot.w_274_3 ctx), ((generatedRoot.w_274_4 ctx), ((generatedRoot.w_274_5 ctx), ())))))), w⟩ : parallel_generatedRoot_275.Ctx backend hashModel params i) (((generatedRoot.w_275_0 ctx) i), (((generatedRoot.w_275_1 ctx) i), (((generatedRoot.w_275_2 ctx) i), (((generatedRoot.w_275_3 ctx) i), (((generatedRoot.w_275_4 ctx) i), (((generatedRoot.w_275_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h275_5 i
  ⟨w, parallel_generatedRoot_275.facts _ h⟩

theorem generatedRoot.h279_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_279.Witness, parallel_generatedRoot_279.Facts (⟨((generatedRoot.w_278_0 ctx), ((generatedRoot.w_278_1 ctx), ((generatedRoot.w_278_2 ctx), ((generatedRoot.w_278_3 ctx), ((generatedRoot.w_278_4 ctx), ((generatedRoot.w_278_5 ctx), ())))))), w⟩ : parallel_generatedRoot_279.Ctx backend hashModel params i) (((generatedRoot.w_279_0 ctx) i), (((generatedRoot.w_279_1 ctx) i), (((generatedRoot.w_279_2 ctx) i), (((generatedRoot.w_279_3 ctx) i), (((generatedRoot.w_279_4 ctx) i), (((generatedRoot.w_279_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h279_5 i
  ⟨w, parallel_generatedRoot_279.facts _ h⟩

theorem generatedRoot.h283_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_283.Witness, parallel_generatedRoot_283.Facts (⟨((generatedRoot.w_282_0 ctx), ((generatedRoot.w_282_1 ctx), ((generatedRoot.w_282_2 ctx), ((generatedRoot.w_282_3 ctx), ((generatedRoot.w_282_4 ctx), ((generatedRoot.w_282_5 ctx), ())))))), w⟩ : parallel_generatedRoot_283.Ctx backend hashModel params i) (((generatedRoot.w_283_0 ctx) i), (((generatedRoot.w_283_1 ctx) i), (((generatedRoot.w_283_2 ctx) i), (((generatedRoot.w_283_3 ctx) i), (((generatedRoot.w_283_4 ctx) i), (((generatedRoot.w_283_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h283_5 i
  ⟨w, parallel_generatedRoot_283.facts _ h⟩

theorem generatedRoot.h287_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_287.Witness, parallel_generatedRoot_287.Facts (⟨((generatedRoot.w_286_0 ctx), ((generatedRoot.w_286_1 ctx), ((generatedRoot.w_286_2 ctx), ((generatedRoot.w_286_3 ctx), ((generatedRoot.w_286_4 ctx), ((generatedRoot.w_286_5 ctx), ())))))), w⟩ : parallel_generatedRoot_287.Ctx backend hashModel params i) (((generatedRoot.w_287_0 ctx) i), (((generatedRoot.w_287_1 ctx) i), (((generatedRoot.w_287_2 ctx) i), (((generatedRoot.w_287_3 ctx) i), (((generatedRoot.w_287_4 ctx) i), (((generatedRoot.w_287_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h287_5 i
  ⟨w, parallel_generatedRoot_287.facts _ h⟩

theorem generatedRoot.h291_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_291.Witness, parallel_generatedRoot_291.Facts (⟨((generatedRoot.w_290_0 ctx), ((generatedRoot.w_290_1 ctx), ((generatedRoot.w_290_2 ctx), ((generatedRoot.w_290_3 ctx), ((generatedRoot.w_290_4 ctx), ((generatedRoot.w_290_5 ctx), ())))))), w⟩ : parallel_generatedRoot_291.Ctx backend hashModel params i) (((generatedRoot.w_291_0 ctx) i), (((generatedRoot.w_291_1 ctx) i), (((generatedRoot.w_291_2 ctx) i), (((generatedRoot.w_291_3 ctx) i), (((generatedRoot.w_291_4 ctx) i), (((generatedRoot.w_291_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h291_5 i
  ⟨w, parallel_generatedRoot_291.facts _ h⟩

theorem generatedRoot.h295_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_295.Witness, parallel_generatedRoot_295.Facts (⟨((generatedRoot.w_294_0 ctx), ((generatedRoot.w_294_1 ctx), ((generatedRoot.w_294_2 ctx), ((generatedRoot.w_294_3 ctx), ((generatedRoot.w_294_4 ctx), ((generatedRoot.w_294_5 ctx), ())))))), w⟩ : parallel_generatedRoot_295.Ctx backend hashModel params i) (((generatedRoot.w_295_0 ctx) i), (((generatedRoot.w_295_1 ctx) i), (((generatedRoot.w_295_2 ctx) i), (((generatedRoot.w_295_3 ctx) i), (((generatedRoot.w_295_4 ctx) i), (((generatedRoot.w_295_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h295_5 i
  ⟨w, parallel_generatedRoot_295.facts _ h⟩

theorem generatedRoot.h299_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_299.Witness, parallel_generatedRoot_299.Facts (⟨((generatedRoot.w_298_0 ctx), ((generatedRoot.w_298_1 ctx), ((generatedRoot.w_298_2 ctx), ((generatedRoot.w_298_3 ctx), ((generatedRoot.w_298_4 ctx), ((generatedRoot.w_298_5 ctx), ())))))), w⟩ : parallel_generatedRoot_299.Ctx backend hashModel params i) (((generatedRoot.w_299_0 ctx) i), (((generatedRoot.w_299_1 ctx) i), (((generatedRoot.w_299_2 ctx) i), (((generatedRoot.w_299_3 ctx) i), (((generatedRoot.w_299_4 ctx) i), (((generatedRoot.w_299_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h299_5 i
  ⟨w, parallel_generatedRoot_299.facts _ h⟩

theorem generatedRoot.h303_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_303.Witness, parallel_generatedRoot_303.Facts (⟨((generatedRoot.w_302_0 ctx), ((generatedRoot.w_302_1 ctx), ((generatedRoot.w_302_2 ctx), ((generatedRoot.w_302_3 ctx), ((generatedRoot.w_302_4 ctx), ((generatedRoot.w_302_5 ctx), ())))))), w⟩ : parallel_generatedRoot_303.Ctx backend hashModel params i) (((generatedRoot.w_303_0 ctx) i), (((generatedRoot.w_303_1 ctx) i), (((generatedRoot.w_303_2 ctx) i), (((generatedRoot.w_303_3 ctx) i), (((generatedRoot.w_303_4 ctx) i), (((generatedRoot.w_303_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h303_5 i
  ⟨w, parallel_generatedRoot_303.facts _ h⟩

theorem generatedRoot.h307_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_307.Witness, parallel_generatedRoot_307.Facts (⟨((generatedRoot.w_306_0 ctx), ((generatedRoot.w_306_1 ctx), ((generatedRoot.w_306_2 ctx), ((generatedRoot.w_306_3 ctx), ((generatedRoot.w_306_4 ctx), ((generatedRoot.w_306_5 ctx), ())))))), w⟩ : parallel_generatedRoot_307.Ctx backend hashModel params i) (((generatedRoot.w_307_0 ctx) i), (((generatedRoot.w_307_1 ctx) i), (((generatedRoot.w_307_2 ctx) i), (((generatedRoot.w_307_3 ctx) i), (((generatedRoot.w_307_4 ctx) i), (((generatedRoot.w_307_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h307_5 i
  ⟨w, parallel_generatedRoot_307.facts _ h⟩

theorem generatedRoot.h311_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_311.Witness, parallel_generatedRoot_311.Facts (⟨((generatedRoot.w_310_0 ctx), ((generatedRoot.w_310_1 ctx), ((generatedRoot.w_310_2 ctx), ((generatedRoot.w_310_3 ctx), ((generatedRoot.w_310_4 ctx), ((generatedRoot.w_310_5 ctx), ())))))), w⟩ : parallel_generatedRoot_311.Ctx backend hashModel params i) (((generatedRoot.w_311_0 ctx) i), (((generatedRoot.w_311_1 ctx) i), (((generatedRoot.w_311_2 ctx) i), (((generatedRoot.w_311_3 ctx) i), (((generatedRoot.w_311_4 ctx) i), (((generatedRoot.w_311_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h311_5 i
  ⟨w, parallel_generatedRoot_311.facts _ h⟩

theorem generatedRoot.h315_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_315.Witness, parallel_generatedRoot_315.Facts (⟨((generatedRoot.w_314_0 ctx), ((generatedRoot.w_314_1 ctx), ((generatedRoot.w_314_2 ctx), ((generatedRoot.w_314_3 ctx), ((generatedRoot.w_314_4 ctx), ((generatedRoot.w_314_5 ctx), ())))))), w⟩ : parallel_generatedRoot_315.Ctx backend hashModel params i) (((generatedRoot.w_315_0 ctx) i), (((generatedRoot.w_315_1 ctx) i), (((generatedRoot.w_315_2 ctx) i), (((generatedRoot.w_315_3 ctx) i), (((generatedRoot.w_315_4 ctx) i), (((generatedRoot.w_315_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h315_5 i
  ⟨w, parallel_generatedRoot_315.facts _ h⟩

theorem generatedRoot.h316_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_316.Witness, parallel_generatedRoot_316.Facts (⟨((generatedRoot.w_134_0 ctx), ((generatedRoot.w_135_0 ctx), ((generatedRoot.w_136_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_138_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_139_0 ctx), (((generatedRoot.w_315_4 ctx) i), (((generatedRoot.w_315_0 ctx) i), (((generatedRoot.w_315_2 ctx) i), (((generatedRoot.w_315_5 ctx) i), (((generatedRoot.w_315_1 ctx) i), (((generatedRoot.w_315_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_316.Ctx backend hashModel params i) (((generatedRoot.w_316_0 ctx) i), (((generatedRoot.w_316_1 ctx) i), (((generatedRoot.w_316_2 ctx) i), (((generatedRoot.w_316_3 ctx) i), (((generatedRoot.w_316_4 ctx) i), (((generatedRoot.w_316_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h316_5 i
  ⟨w, parallel_generatedRoot_316.facts _ h⟩

end

end Stage_online_1

end
