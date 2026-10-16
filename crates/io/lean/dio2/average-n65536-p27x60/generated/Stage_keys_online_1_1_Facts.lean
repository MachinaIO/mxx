import Stage_keys_online_1_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_keys_online_1_1

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
structure parallel_generatedRoot_177.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_177.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_177Ctx" => (parallel_generatedRoot_177.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_177.w_0_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_177.w_1_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_177.w_2_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_177.w_3_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_177.w_4_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_177.w_5_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_177.w_6_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_177.w_6_1  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_177.w_6_2  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_177.w_6_3  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_177.w_6_4  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_177.w_6_5  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_177.w_7_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_177.w_8_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_177.w_9_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_7_0 ctx) (parallel_generatedRoot_177.w_8_0 ctx)

def parallel_generatedRoot_177.w_10_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_177.w_11_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_177.w_12_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_10_0 ctx) (parallel_generatedRoot_177.w_11_0 ctx)

def parallel_generatedRoot_177.w_13_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_177.w_14_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_177.w_15_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_13_0 ctx) (parallel_generatedRoot_177.w_14_0 ctx)

def parallel_generatedRoot_177.w_16_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_177.w_17_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_177.w_18_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_16_0 ctx) (parallel_generatedRoot_177.w_17_0 ctx)

def parallel_generatedRoot_177.w_19_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_177.w_20_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_177.w_21_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_19_0 ctx) (parallel_generatedRoot_177.w_20_0 ctx)

def parallel_generatedRoot_177.w_22_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_177.w_23_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_177.w_24_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_22_0 ctx) (parallel_generatedRoot_177.w_23_0 ctx)

def parallel_generatedRoot_177.w_25_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_177.w_25_1  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_177.w_25_2  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_177.w_25_3  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_177.w_25_4  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_177.w_25_5  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_177.outs  (ctx : parallel_generatedRoot_177Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_177.w_25_0 ctx), ((parallel_generatedRoot_177.w_25_1 ctx), ((parallel_generatedRoot_177.w_25_2 ctx), ((parallel_generatedRoot_177.w_25_3 ctx), ((parallel_generatedRoot_177.w_25_4 ctx), ((parallel_generatedRoot_177.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_177.Facts_0  (ctx : parallel_generatedRoot_177Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_177_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_177.w_0_0 ctx), ((parallel_generatedRoot_177.w_1_0 ctx), ((parallel_generatedRoot_177.w_2_0 ctx), ((parallel_generatedRoot_177.w_3_0 ctx), ((parallel_generatedRoot_177.w_4_0 ctx), ((parallel_generatedRoot_177.w_5_0 ctx), ())))))) (((parallel_generatedRoot_177.w_6_0 ctx) i), (((parallel_generatedRoot_177.w_6_1 ctx) i), (((parallel_generatedRoot_177.w_6_2 ctx) i), (((parallel_generatedRoot_177.w_6_3 ctx) i), (((parallel_generatedRoot_177.w_6_4 ctx) i), (((parallel_generatedRoot_177.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_0 ctx) (0) (parallel_generatedRoot_177.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_0 ctx) (1) (parallel_generatedRoot_177.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_1 ctx) (0) (parallel_generatedRoot_177.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_1 ctx) (1) (parallel_generatedRoot_177.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_2 ctx) (0) (parallel_generatedRoot_177.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_2 ctx) (1) (parallel_generatedRoot_177.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_3 ctx) (0) (parallel_generatedRoot_177.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_3 ctx) (1) (parallel_generatedRoot_177.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_4 ctx) (0) (parallel_generatedRoot_177.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_4 ctx) (1) (parallel_generatedRoot_177.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_5 ctx) (0) (parallel_generatedRoot_177.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_6_5 ctx) (1) (parallel_generatedRoot_177.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_177_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_177.w_6_0 ctx), ((parallel_generatedRoot_177.w_6_1 ctx), ((parallel_generatedRoot_177.w_6_2 ctx), ((parallel_generatedRoot_177.w_6_3 ctx), ((parallel_generatedRoot_177.w_6_4 ctx), ((parallel_generatedRoot_177.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_177.w_9_0 ctx), ((parallel_generatedRoot_177.w_12_0 ctx), ((parallel_generatedRoot_177.w_15_0 ctx), ((parallel_generatedRoot_177.w_18_0 ctx), ((parallel_generatedRoot_177.w_21_0 ctx), ((parallel_generatedRoot_177.w_24_0 ctx), ())))))) ((parallel_generatedRoot_177.w_25_0 ctx), ((parallel_generatedRoot_177.w_25_1 ctx), ((parallel_generatedRoot_177.w_25_2 ctx), ((parallel_generatedRoot_177.w_25_3 ctx), ((parallel_generatedRoot_177.w_25_4 ctx), ((parallel_generatedRoot_177.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_177.outs ctx

theorem parallel_generatedRoot_177.facts_0  {ctx : parallel_generatedRoot_177Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_177.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_177.w_0_0 ctx) (parallel_generatedRoot_177.w_1_0 ctx) (parallel_generatedRoot_177.w_2_0 ctx) (parallel_generatedRoot_177.w_3_0 ctx) (parallel_generatedRoot_177.w_4_0 ctx) (parallel_generatedRoot_177.w_5_0 ctx) (parallel_generatedRoot_177.w_6_0 ctx) (parallel_generatedRoot_177.w_6_1 ctx) (parallel_generatedRoot_177.w_6_2 ctx) (parallel_generatedRoot_177.w_6_3 ctx) (parallel_generatedRoot_177.w_6_4 ctx) (parallel_generatedRoot_177.w_6_5 ctx) (parallel_generatedRoot_177.w_7_0 ctx) (parallel_generatedRoot_177.w_8_0 ctx) (parallel_generatedRoot_177.w_9_0 ctx) (parallel_generatedRoot_177.w_10_0 ctx) (parallel_generatedRoot_177.w_11_0 ctx) (parallel_generatedRoot_177.w_12_0 ctx) (parallel_generatedRoot_177.w_13_0 ctx) (parallel_generatedRoot_177.w_14_0 ctx) (parallel_generatedRoot_177.w_15_0 ctx) (parallel_generatedRoot_177.w_16_0 ctx) (parallel_generatedRoot_177.w_17_0 ctx) (parallel_generatedRoot_177.w_18_0 ctx) (parallel_generatedRoot_177.w_19_0 ctx) (parallel_generatedRoot_177.w_20_0 ctx) (parallel_generatedRoot_177.w_21_0 ctx) (parallel_generatedRoot_177.w_22_0 ctx) (parallel_generatedRoot_177.w_23_0 ctx) (parallel_generatedRoot_177.w_24_0 ctx) (parallel_generatedRoot_177.w_25_0 ctx) (parallel_generatedRoot_177.w_25_1 ctx) (parallel_generatedRoot_177.w_25_2 ctx) (parallel_generatedRoot_177.w_25_3 ctx) (parallel_generatedRoot_177.w_25_4 ctx) (parallel_generatedRoot_177.w_25_5 ctx) outputs) :
    parallel_generatedRoot_177.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_177.Facts  (ctx : parallel_generatedRoot_177Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_177.Facts_0 ctx outputs

theorem parallel_generatedRoot_177.facts  (ctx : parallel_generatedRoot_177Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_177.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_177.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_177.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_181.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_181.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_181Ctx" => (parallel_generatedRoot_181.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_181.w_0_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_181.w_1_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_181.w_2_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_181.w_3_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_181.w_4_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_181.w_5_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_181.w_6_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_181.w_6_1  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_181.w_6_2  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_181.w_6_3  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_181.w_6_4  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_181.w_6_5  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_181.w_7_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_181.w_8_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_181.w_9_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_7_0 ctx) (parallel_generatedRoot_181.w_8_0 ctx)

def parallel_generatedRoot_181.w_10_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_181.w_11_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_181.w_12_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_10_0 ctx) (parallel_generatedRoot_181.w_11_0 ctx)

def parallel_generatedRoot_181.w_13_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_181.w_14_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_181.w_15_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_13_0 ctx) (parallel_generatedRoot_181.w_14_0 ctx)

def parallel_generatedRoot_181.w_16_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_181.w_17_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_181.w_18_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_16_0 ctx) (parallel_generatedRoot_181.w_17_0 ctx)

def parallel_generatedRoot_181.w_19_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_181.w_20_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_181.w_21_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_19_0 ctx) (parallel_generatedRoot_181.w_20_0 ctx)

def parallel_generatedRoot_181.w_22_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_181.w_23_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_181.w_24_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_22_0 ctx) (parallel_generatedRoot_181.w_23_0 ctx)

def parallel_generatedRoot_181.w_25_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_181.w_25_1  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_181.w_25_2  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_181.w_25_3  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_181.w_25_4  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_181.w_25_5  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_181.outs  (ctx : parallel_generatedRoot_181Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_181.w_25_0 ctx), ((parallel_generatedRoot_181.w_25_1 ctx), ((parallel_generatedRoot_181.w_25_2 ctx), ((parallel_generatedRoot_181.w_25_3 ctx), ((parallel_generatedRoot_181.w_25_4 ctx), ((parallel_generatedRoot_181.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_181.Facts_0  (ctx : parallel_generatedRoot_181Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_181_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_181.w_0_0 ctx), ((parallel_generatedRoot_181.w_1_0 ctx), ((parallel_generatedRoot_181.w_2_0 ctx), ((parallel_generatedRoot_181.w_3_0 ctx), ((parallel_generatedRoot_181.w_4_0 ctx), ((parallel_generatedRoot_181.w_5_0 ctx), ())))))) (((parallel_generatedRoot_181.w_6_0 ctx) i), (((parallel_generatedRoot_181.w_6_1 ctx) i), (((parallel_generatedRoot_181.w_6_2 ctx) i), (((parallel_generatedRoot_181.w_6_3 ctx) i), (((parallel_generatedRoot_181.w_6_4 ctx) i), (((parallel_generatedRoot_181.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_0 ctx) (0) (parallel_generatedRoot_181.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_0 ctx) (1) (parallel_generatedRoot_181.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_1 ctx) (0) (parallel_generatedRoot_181.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_1 ctx) (1) (parallel_generatedRoot_181.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_2 ctx) (0) (parallel_generatedRoot_181.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_2 ctx) (1) (parallel_generatedRoot_181.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_3 ctx) (0) (parallel_generatedRoot_181.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_3 ctx) (1) (parallel_generatedRoot_181.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_4 ctx) (0) (parallel_generatedRoot_181.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_4 ctx) (1) (parallel_generatedRoot_181.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_5 ctx) (0) (parallel_generatedRoot_181.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_6_5 ctx) (1) (parallel_generatedRoot_181.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_181_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_181.w_6_0 ctx), ((parallel_generatedRoot_181.w_6_1 ctx), ((parallel_generatedRoot_181.w_6_2 ctx), ((parallel_generatedRoot_181.w_6_3 ctx), ((parallel_generatedRoot_181.w_6_4 ctx), ((parallel_generatedRoot_181.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_181.w_9_0 ctx), ((parallel_generatedRoot_181.w_12_0 ctx), ((parallel_generatedRoot_181.w_15_0 ctx), ((parallel_generatedRoot_181.w_18_0 ctx), ((parallel_generatedRoot_181.w_21_0 ctx), ((parallel_generatedRoot_181.w_24_0 ctx), ())))))) ((parallel_generatedRoot_181.w_25_0 ctx), ((parallel_generatedRoot_181.w_25_1 ctx), ((parallel_generatedRoot_181.w_25_2 ctx), ((parallel_generatedRoot_181.w_25_3 ctx), ((parallel_generatedRoot_181.w_25_4 ctx), ((parallel_generatedRoot_181.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_181.outs ctx

theorem parallel_generatedRoot_181.facts_0  {ctx : parallel_generatedRoot_181Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_181.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_181.w_0_0 ctx) (parallel_generatedRoot_181.w_1_0 ctx) (parallel_generatedRoot_181.w_2_0 ctx) (parallel_generatedRoot_181.w_3_0 ctx) (parallel_generatedRoot_181.w_4_0 ctx) (parallel_generatedRoot_181.w_5_0 ctx) (parallel_generatedRoot_181.w_6_0 ctx) (parallel_generatedRoot_181.w_6_1 ctx) (parallel_generatedRoot_181.w_6_2 ctx) (parallel_generatedRoot_181.w_6_3 ctx) (parallel_generatedRoot_181.w_6_4 ctx) (parallel_generatedRoot_181.w_6_5 ctx) (parallel_generatedRoot_181.w_7_0 ctx) (parallel_generatedRoot_181.w_8_0 ctx) (parallel_generatedRoot_181.w_9_0 ctx) (parallel_generatedRoot_181.w_10_0 ctx) (parallel_generatedRoot_181.w_11_0 ctx) (parallel_generatedRoot_181.w_12_0 ctx) (parallel_generatedRoot_181.w_13_0 ctx) (parallel_generatedRoot_181.w_14_0 ctx) (parallel_generatedRoot_181.w_15_0 ctx) (parallel_generatedRoot_181.w_16_0 ctx) (parallel_generatedRoot_181.w_17_0 ctx) (parallel_generatedRoot_181.w_18_0 ctx) (parallel_generatedRoot_181.w_19_0 ctx) (parallel_generatedRoot_181.w_20_0 ctx) (parallel_generatedRoot_181.w_21_0 ctx) (parallel_generatedRoot_181.w_22_0 ctx) (parallel_generatedRoot_181.w_23_0 ctx) (parallel_generatedRoot_181.w_24_0 ctx) (parallel_generatedRoot_181.w_25_0 ctx) (parallel_generatedRoot_181.w_25_1 ctx) (parallel_generatedRoot_181.w_25_2 ctx) (parallel_generatedRoot_181.w_25_3 ctx) (parallel_generatedRoot_181.w_25_4 ctx) (parallel_generatedRoot_181.w_25_5 ctx) outputs) :
    parallel_generatedRoot_181.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_181.Facts  (ctx : parallel_generatedRoot_181Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_181.Facts_0 ctx outputs

theorem parallel_generatedRoot_181.facts  (ctx : parallel_generatedRoot_181Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_181.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_181.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_181.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_185.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_185.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_185Ctx" => (parallel_generatedRoot_185.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_185.w_0_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_185.w_1_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_185.w_2_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_185.w_3_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_185.w_4_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_185.w_5_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_185.w_6_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_185.w_6_1  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_185.w_6_2  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_185.w_6_3  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_185.w_6_4  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_185.w_6_5  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_185.w_7_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_185.w_8_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_185.w_9_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_7_0 ctx) (parallel_generatedRoot_185.w_8_0 ctx)

def parallel_generatedRoot_185.w_10_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_185.w_11_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_185.w_12_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_10_0 ctx) (parallel_generatedRoot_185.w_11_0 ctx)

def parallel_generatedRoot_185.w_13_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_185.w_14_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_185.w_15_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_13_0 ctx) (parallel_generatedRoot_185.w_14_0 ctx)

def parallel_generatedRoot_185.w_16_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_185.w_17_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_185.w_18_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_16_0 ctx) (parallel_generatedRoot_185.w_17_0 ctx)

def parallel_generatedRoot_185.w_19_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_185.w_20_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_185.w_21_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_19_0 ctx) (parallel_generatedRoot_185.w_20_0 ctx)

def parallel_generatedRoot_185.w_22_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_185.w_23_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_185.w_24_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_22_0 ctx) (parallel_generatedRoot_185.w_23_0 ctx)

def parallel_generatedRoot_185.w_25_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_185.w_25_1  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_185.w_25_2  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_185.w_25_3  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_185.w_25_4  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_185.w_25_5  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_185.outs  (ctx : parallel_generatedRoot_185Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_185.w_25_0 ctx), ((parallel_generatedRoot_185.w_25_1 ctx), ((parallel_generatedRoot_185.w_25_2 ctx), ((parallel_generatedRoot_185.w_25_3 ctx), ((parallel_generatedRoot_185.w_25_4 ctx), ((parallel_generatedRoot_185.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_185.Facts_0  (ctx : parallel_generatedRoot_185Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_185_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_185.w_0_0 ctx), ((parallel_generatedRoot_185.w_1_0 ctx), ((parallel_generatedRoot_185.w_2_0 ctx), ((parallel_generatedRoot_185.w_3_0 ctx), ((parallel_generatedRoot_185.w_4_0 ctx), ((parallel_generatedRoot_185.w_5_0 ctx), ())))))) (((parallel_generatedRoot_185.w_6_0 ctx) i), (((parallel_generatedRoot_185.w_6_1 ctx) i), (((parallel_generatedRoot_185.w_6_2 ctx) i), (((parallel_generatedRoot_185.w_6_3 ctx) i), (((parallel_generatedRoot_185.w_6_4 ctx) i), (((parallel_generatedRoot_185.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_0 ctx) (0) (parallel_generatedRoot_185.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_0 ctx) (1) (parallel_generatedRoot_185.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_1 ctx) (0) (parallel_generatedRoot_185.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_1 ctx) (1) (parallel_generatedRoot_185.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_2 ctx) (0) (parallel_generatedRoot_185.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_2 ctx) (1) (parallel_generatedRoot_185.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_3 ctx) (0) (parallel_generatedRoot_185.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_3 ctx) (1) (parallel_generatedRoot_185.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_4 ctx) (0) (parallel_generatedRoot_185.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_4 ctx) (1) (parallel_generatedRoot_185.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_5 ctx) (0) (parallel_generatedRoot_185.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_6_5 ctx) (1) (parallel_generatedRoot_185.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_185_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_185.w_6_0 ctx), ((parallel_generatedRoot_185.w_6_1 ctx), ((parallel_generatedRoot_185.w_6_2 ctx), ((parallel_generatedRoot_185.w_6_3 ctx), ((parallel_generatedRoot_185.w_6_4 ctx), ((parallel_generatedRoot_185.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_185.w_9_0 ctx), ((parallel_generatedRoot_185.w_12_0 ctx), ((parallel_generatedRoot_185.w_15_0 ctx), ((parallel_generatedRoot_185.w_18_0 ctx), ((parallel_generatedRoot_185.w_21_0 ctx), ((parallel_generatedRoot_185.w_24_0 ctx), ())))))) ((parallel_generatedRoot_185.w_25_0 ctx), ((parallel_generatedRoot_185.w_25_1 ctx), ((parallel_generatedRoot_185.w_25_2 ctx), ((parallel_generatedRoot_185.w_25_3 ctx), ((parallel_generatedRoot_185.w_25_4 ctx), ((parallel_generatedRoot_185.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_185.outs ctx

theorem parallel_generatedRoot_185.facts_0  {ctx : parallel_generatedRoot_185Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_185.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_185.w_0_0 ctx) (parallel_generatedRoot_185.w_1_0 ctx) (parallel_generatedRoot_185.w_2_0 ctx) (parallel_generatedRoot_185.w_3_0 ctx) (parallel_generatedRoot_185.w_4_0 ctx) (parallel_generatedRoot_185.w_5_0 ctx) (parallel_generatedRoot_185.w_6_0 ctx) (parallel_generatedRoot_185.w_6_1 ctx) (parallel_generatedRoot_185.w_6_2 ctx) (parallel_generatedRoot_185.w_6_3 ctx) (parallel_generatedRoot_185.w_6_4 ctx) (parallel_generatedRoot_185.w_6_5 ctx) (parallel_generatedRoot_185.w_7_0 ctx) (parallel_generatedRoot_185.w_8_0 ctx) (parallel_generatedRoot_185.w_9_0 ctx) (parallel_generatedRoot_185.w_10_0 ctx) (parallel_generatedRoot_185.w_11_0 ctx) (parallel_generatedRoot_185.w_12_0 ctx) (parallel_generatedRoot_185.w_13_0 ctx) (parallel_generatedRoot_185.w_14_0 ctx) (parallel_generatedRoot_185.w_15_0 ctx) (parallel_generatedRoot_185.w_16_0 ctx) (parallel_generatedRoot_185.w_17_0 ctx) (parallel_generatedRoot_185.w_18_0 ctx) (parallel_generatedRoot_185.w_19_0 ctx) (parallel_generatedRoot_185.w_20_0 ctx) (parallel_generatedRoot_185.w_21_0 ctx) (parallel_generatedRoot_185.w_22_0 ctx) (parallel_generatedRoot_185.w_23_0 ctx) (parallel_generatedRoot_185.w_24_0 ctx) (parallel_generatedRoot_185.w_25_0 ctx) (parallel_generatedRoot_185.w_25_1 ctx) (parallel_generatedRoot_185.w_25_2 ctx) (parallel_generatedRoot_185.w_25_3 ctx) (parallel_generatedRoot_185.w_25_4 ctx) (parallel_generatedRoot_185.w_25_5 ctx) outputs) :
    parallel_generatedRoot_185.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_185.Facts  (ctx : parallel_generatedRoot_185Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_185.Facts_0 ctx outputs

theorem parallel_generatedRoot_185.facts  (ctx : parallel_generatedRoot_185Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_185.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_185.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_185.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_189.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_189.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_189Ctx" => (parallel_generatedRoot_189.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_189.w_0_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_189.w_1_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_189.w_2_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_189.w_3_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_189.w_4_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_189.w_5_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_189.w_6_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_189.w_6_1  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_189.w_6_2  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_189.w_6_3  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_189.w_6_4  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_189.w_6_5  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_189.w_7_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_189.w_8_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_189.w_9_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_7_0 ctx) (parallel_generatedRoot_189.w_8_0 ctx)

def parallel_generatedRoot_189.w_10_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_189.w_11_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_189.w_12_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_10_0 ctx) (parallel_generatedRoot_189.w_11_0 ctx)

def parallel_generatedRoot_189.w_13_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_189.w_14_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_189.w_15_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_13_0 ctx) (parallel_generatedRoot_189.w_14_0 ctx)

def parallel_generatedRoot_189.w_16_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_189.w_17_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_189.w_18_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_16_0 ctx) (parallel_generatedRoot_189.w_17_0 ctx)

def parallel_generatedRoot_189.w_19_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_189.w_20_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_189.w_21_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_19_0 ctx) (parallel_generatedRoot_189.w_20_0 ctx)

def parallel_generatedRoot_189.w_22_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_189.w_23_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_189.w_24_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_22_0 ctx) (parallel_generatedRoot_189.w_23_0 ctx)

def parallel_generatedRoot_189.w_25_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_189.w_25_1  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_189.w_25_2  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_189.w_25_3  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_189.w_25_4  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_189.w_25_5  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_189.outs  (ctx : parallel_generatedRoot_189Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_189.w_25_0 ctx), ((parallel_generatedRoot_189.w_25_1 ctx), ((parallel_generatedRoot_189.w_25_2 ctx), ((parallel_generatedRoot_189.w_25_3 ctx), ((parallel_generatedRoot_189.w_25_4 ctx), ((parallel_generatedRoot_189.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_189.Facts_0  (ctx : parallel_generatedRoot_189Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_189_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_189.w_0_0 ctx), ((parallel_generatedRoot_189.w_1_0 ctx), ((parallel_generatedRoot_189.w_2_0 ctx), ((parallel_generatedRoot_189.w_3_0 ctx), ((parallel_generatedRoot_189.w_4_0 ctx), ((parallel_generatedRoot_189.w_5_0 ctx), ())))))) (((parallel_generatedRoot_189.w_6_0 ctx) i), (((parallel_generatedRoot_189.w_6_1 ctx) i), (((parallel_generatedRoot_189.w_6_2 ctx) i), (((parallel_generatedRoot_189.w_6_3 ctx) i), (((parallel_generatedRoot_189.w_6_4 ctx) i), (((parallel_generatedRoot_189.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_0 ctx) (0) (parallel_generatedRoot_189.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_0 ctx) (1) (parallel_generatedRoot_189.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_1 ctx) (0) (parallel_generatedRoot_189.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_1 ctx) (1) (parallel_generatedRoot_189.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_2 ctx) (0) (parallel_generatedRoot_189.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_2 ctx) (1) (parallel_generatedRoot_189.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_3 ctx) (0) (parallel_generatedRoot_189.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_3 ctx) (1) (parallel_generatedRoot_189.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_4 ctx) (0) (parallel_generatedRoot_189.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_4 ctx) (1) (parallel_generatedRoot_189.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_5 ctx) (0) (parallel_generatedRoot_189.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_6_5 ctx) (1) (parallel_generatedRoot_189.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_189_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_189.w_6_0 ctx), ((parallel_generatedRoot_189.w_6_1 ctx), ((parallel_generatedRoot_189.w_6_2 ctx), ((parallel_generatedRoot_189.w_6_3 ctx), ((parallel_generatedRoot_189.w_6_4 ctx), ((parallel_generatedRoot_189.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_189.w_9_0 ctx), ((parallel_generatedRoot_189.w_12_0 ctx), ((parallel_generatedRoot_189.w_15_0 ctx), ((parallel_generatedRoot_189.w_18_0 ctx), ((parallel_generatedRoot_189.w_21_0 ctx), ((parallel_generatedRoot_189.w_24_0 ctx), ())))))) ((parallel_generatedRoot_189.w_25_0 ctx), ((parallel_generatedRoot_189.w_25_1 ctx), ((parallel_generatedRoot_189.w_25_2 ctx), ((parallel_generatedRoot_189.w_25_3 ctx), ((parallel_generatedRoot_189.w_25_4 ctx), ((parallel_generatedRoot_189.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_189.outs ctx

theorem parallel_generatedRoot_189.facts_0  {ctx : parallel_generatedRoot_189Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_189.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_189.w_0_0 ctx) (parallel_generatedRoot_189.w_1_0 ctx) (parallel_generatedRoot_189.w_2_0 ctx) (parallel_generatedRoot_189.w_3_0 ctx) (parallel_generatedRoot_189.w_4_0 ctx) (parallel_generatedRoot_189.w_5_0 ctx) (parallel_generatedRoot_189.w_6_0 ctx) (parallel_generatedRoot_189.w_6_1 ctx) (parallel_generatedRoot_189.w_6_2 ctx) (parallel_generatedRoot_189.w_6_3 ctx) (parallel_generatedRoot_189.w_6_4 ctx) (parallel_generatedRoot_189.w_6_5 ctx) (parallel_generatedRoot_189.w_7_0 ctx) (parallel_generatedRoot_189.w_8_0 ctx) (parallel_generatedRoot_189.w_9_0 ctx) (parallel_generatedRoot_189.w_10_0 ctx) (parallel_generatedRoot_189.w_11_0 ctx) (parallel_generatedRoot_189.w_12_0 ctx) (parallel_generatedRoot_189.w_13_0 ctx) (parallel_generatedRoot_189.w_14_0 ctx) (parallel_generatedRoot_189.w_15_0 ctx) (parallel_generatedRoot_189.w_16_0 ctx) (parallel_generatedRoot_189.w_17_0 ctx) (parallel_generatedRoot_189.w_18_0 ctx) (parallel_generatedRoot_189.w_19_0 ctx) (parallel_generatedRoot_189.w_20_0 ctx) (parallel_generatedRoot_189.w_21_0 ctx) (parallel_generatedRoot_189.w_22_0 ctx) (parallel_generatedRoot_189.w_23_0 ctx) (parallel_generatedRoot_189.w_24_0 ctx) (parallel_generatedRoot_189.w_25_0 ctx) (parallel_generatedRoot_189.w_25_1 ctx) (parallel_generatedRoot_189.w_25_2 ctx) (parallel_generatedRoot_189.w_25_3 ctx) (parallel_generatedRoot_189.w_25_4 ctx) (parallel_generatedRoot_189.w_25_5 ctx) outputs) :
    parallel_generatedRoot_189.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_189.Facts  (ctx : parallel_generatedRoot_189Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_189.Facts_0 ctx outputs

theorem parallel_generatedRoot_189.facts  (ctx : parallel_generatedRoot_189Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_189.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_189.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_189.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_193.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_193.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_193Ctx" => (parallel_generatedRoot_193.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_193.w_0_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_193.w_1_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_193.w_2_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_193.w_3_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_193.w_4_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_193.w_5_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_193.w_6_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_193.w_6_1  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_193.w_6_2  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_193.w_6_3  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_193.w_6_4  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_193.w_6_5  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_193.w_7_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_193.w_8_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_193.w_9_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_7_0 ctx) (parallel_generatedRoot_193.w_8_0 ctx)

def parallel_generatedRoot_193.w_10_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_193.w_11_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_193.w_12_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_10_0 ctx) (parallel_generatedRoot_193.w_11_0 ctx)

def parallel_generatedRoot_193.w_13_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_193.w_14_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_193.w_15_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_13_0 ctx) (parallel_generatedRoot_193.w_14_0 ctx)

def parallel_generatedRoot_193.w_16_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_193.w_17_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_193.w_18_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_16_0 ctx) (parallel_generatedRoot_193.w_17_0 ctx)

def parallel_generatedRoot_193.w_19_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_193.w_20_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_193.w_21_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_19_0 ctx) (parallel_generatedRoot_193.w_20_0 ctx)

def parallel_generatedRoot_193.w_22_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_193.w_23_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_193.w_24_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_22_0 ctx) (parallel_generatedRoot_193.w_23_0 ctx)

def parallel_generatedRoot_193.w_25_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_193.w_25_1  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_193.w_25_2  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_193.w_25_3  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_193.w_25_4  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_193.w_25_5  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_193.outs  (ctx : parallel_generatedRoot_193Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_193.w_25_0 ctx), ((parallel_generatedRoot_193.w_25_1 ctx), ((parallel_generatedRoot_193.w_25_2 ctx), ((parallel_generatedRoot_193.w_25_3 ctx), ((parallel_generatedRoot_193.w_25_4 ctx), ((parallel_generatedRoot_193.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_193.Facts_0  (ctx : parallel_generatedRoot_193Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_193_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_193.w_0_0 ctx), ((parallel_generatedRoot_193.w_1_0 ctx), ((parallel_generatedRoot_193.w_2_0 ctx), ((parallel_generatedRoot_193.w_3_0 ctx), ((parallel_generatedRoot_193.w_4_0 ctx), ((parallel_generatedRoot_193.w_5_0 ctx), ())))))) (((parallel_generatedRoot_193.w_6_0 ctx) i), (((parallel_generatedRoot_193.w_6_1 ctx) i), (((parallel_generatedRoot_193.w_6_2 ctx) i), (((parallel_generatedRoot_193.w_6_3 ctx) i), (((parallel_generatedRoot_193.w_6_4 ctx) i), (((parallel_generatedRoot_193.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_0 ctx) (0) (parallel_generatedRoot_193.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_0 ctx) (1) (parallel_generatedRoot_193.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_1 ctx) (0) (parallel_generatedRoot_193.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_1 ctx) (1) (parallel_generatedRoot_193.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_2 ctx) (0) (parallel_generatedRoot_193.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_2 ctx) (1) (parallel_generatedRoot_193.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_3 ctx) (0) (parallel_generatedRoot_193.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_3 ctx) (1) (parallel_generatedRoot_193.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_4 ctx) (0) (parallel_generatedRoot_193.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_4 ctx) (1) (parallel_generatedRoot_193.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_5 ctx) (0) (parallel_generatedRoot_193.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_6_5 ctx) (1) (parallel_generatedRoot_193.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_193_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_193.w_6_0 ctx), ((parallel_generatedRoot_193.w_6_1 ctx), ((parallel_generatedRoot_193.w_6_2 ctx), ((parallel_generatedRoot_193.w_6_3 ctx), ((parallel_generatedRoot_193.w_6_4 ctx), ((parallel_generatedRoot_193.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_193.w_9_0 ctx), ((parallel_generatedRoot_193.w_12_0 ctx), ((parallel_generatedRoot_193.w_15_0 ctx), ((parallel_generatedRoot_193.w_18_0 ctx), ((parallel_generatedRoot_193.w_21_0 ctx), ((parallel_generatedRoot_193.w_24_0 ctx), ())))))) ((parallel_generatedRoot_193.w_25_0 ctx), ((parallel_generatedRoot_193.w_25_1 ctx), ((parallel_generatedRoot_193.w_25_2 ctx), ((parallel_generatedRoot_193.w_25_3 ctx), ((parallel_generatedRoot_193.w_25_4 ctx), ((parallel_generatedRoot_193.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_193.outs ctx

theorem parallel_generatedRoot_193.facts_0  {ctx : parallel_generatedRoot_193Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_193.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_193.w_0_0 ctx) (parallel_generatedRoot_193.w_1_0 ctx) (parallel_generatedRoot_193.w_2_0 ctx) (parallel_generatedRoot_193.w_3_0 ctx) (parallel_generatedRoot_193.w_4_0 ctx) (parallel_generatedRoot_193.w_5_0 ctx) (parallel_generatedRoot_193.w_6_0 ctx) (parallel_generatedRoot_193.w_6_1 ctx) (parallel_generatedRoot_193.w_6_2 ctx) (parallel_generatedRoot_193.w_6_3 ctx) (parallel_generatedRoot_193.w_6_4 ctx) (parallel_generatedRoot_193.w_6_5 ctx) (parallel_generatedRoot_193.w_7_0 ctx) (parallel_generatedRoot_193.w_8_0 ctx) (parallel_generatedRoot_193.w_9_0 ctx) (parallel_generatedRoot_193.w_10_0 ctx) (parallel_generatedRoot_193.w_11_0 ctx) (parallel_generatedRoot_193.w_12_0 ctx) (parallel_generatedRoot_193.w_13_0 ctx) (parallel_generatedRoot_193.w_14_0 ctx) (parallel_generatedRoot_193.w_15_0 ctx) (parallel_generatedRoot_193.w_16_0 ctx) (parallel_generatedRoot_193.w_17_0 ctx) (parallel_generatedRoot_193.w_18_0 ctx) (parallel_generatedRoot_193.w_19_0 ctx) (parallel_generatedRoot_193.w_20_0 ctx) (parallel_generatedRoot_193.w_21_0 ctx) (parallel_generatedRoot_193.w_22_0 ctx) (parallel_generatedRoot_193.w_23_0 ctx) (parallel_generatedRoot_193.w_24_0 ctx) (parallel_generatedRoot_193.w_25_0 ctx) (parallel_generatedRoot_193.w_25_1 ctx) (parallel_generatedRoot_193.w_25_2 ctx) (parallel_generatedRoot_193.w_25_3 ctx) (parallel_generatedRoot_193.w_25_4 ctx) (parallel_generatedRoot_193.w_25_5 ctx) outputs) :
    parallel_generatedRoot_193.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_193.Facts  (ctx : parallel_generatedRoot_193Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_193.Facts_0 ctx outputs

theorem parallel_generatedRoot_193.facts  (ctx : parallel_generatedRoot_193Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_193.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_193.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_193.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_197.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_197.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_197Ctx" => (parallel_generatedRoot_197.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_197.w_0_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_197.w_1_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_197.w_2_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_197.w_3_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_197.w_4_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_197.w_5_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_197.w_6_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_197.w_6_1  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_197.w_6_2  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_197.w_6_3  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_197.w_6_4  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_197.w_6_5  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_197.w_7_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_197.w_8_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_197.w_9_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_7_0 ctx) (parallel_generatedRoot_197.w_8_0 ctx)

def parallel_generatedRoot_197.w_10_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_197.w_11_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_197.w_12_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_10_0 ctx) (parallel_generatedRoot_197.w_11_0 ctx)

def parallel_generatedRoot_197.w_13_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_197.w_14_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_197.w_15_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_13_0 ctx) (parallel_generatedRoot_197.w_14_0 ctx)

def parallel_generatedRoot_197.w_16_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_197.w_17_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_197.w_18_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_16_0 ctx) (parallel_generatedRoot_197.w_17_0 ctx)

def parallel_generatedRoot_197.w_19_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_197.w_20_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_197.w_21_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_19_0 ctx) (parallel_generatedRoot_197.w_20_0 ctx)

def parallel_generatedRoot_197.w_22_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_197.w_23_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_197.w_24_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_22_0 ctx) (parallel_generatedRoot_197.w_23_0 ctx)

def parallel_generatedRoot_197.w_25_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_197.w_25_1  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_197.w_25_2  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_197.w_25_3  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_197.w_25_4  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_197.w_25_5  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_197.outs  (ctx : parallel_generatedRoot_197Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_197.w_25_0 ctx), ((parallel_generatedRoot_197.w_25_1 ctx), ((parallel_generatedRoot_197.w_25_2 ctx), ((parallel_generatedRoot_197.w_25_3 ctx), ((parallel_generatedRoot_197.w_25_4 ctx), ((parallel_generatedRoot_197.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_197.Facts_0  (ctx : parallel_generatedRoot_197Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_197_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_197.w_0_0 ctx), ((parallel_generatedRoot_197.w_1_0 ctx), ((parallel_generatedRoot_197.w_2_0 ctx), ((parallel_generatedRoot_197.w_3_0 ctx), ((parallel_generatedRoot_197.w_4_0 ctx), ((parallel_generatedRoot_197.w_5_0 ctx), ())))))) (((parallel_generatedRoot_197.w_6_0 ctx) i), (((parallel_generatedRoot_197.w_6_1 ctx) i), (((parallel_generatedRoot_197.w_6_2 ctx) i), (((parallel_generatedRoot_197.w_6_3 ctx) i), (((parallel_generatedRoot_197.w_6_4 ctx) i), (((parallel_generatedRoot_197.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_0 ctx) (0) (parallel_generatedRoot_197.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_0 ctx) (1) (parallel_generatedRoot_197.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_1 ctx) (0) (parallel_generatedRoot_197.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_1 ctx) (1) (parallel_generatedRoot_197.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_2 ctx) (0) (parallel_generatedRoot_197.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_2 ctx) (1) (parallel_generatedRoot_197.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_3 ctx) (0) (parallel_generatedRoot_197.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_3 ctx) (1) (parallel_generatedRoot_197.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_4 ctx) (0) (parallel_generatedRoot_197.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_4 ctx) (1) (parallel_generatedRoot_197.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_5 ctx) (0) (parallel_generatedRoot_197.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_6_5 ctx) (1) (parallel_generatedRoot_197.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_197_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_197.w_6_0 ctx), ((parallel_generatedRoot_197.w_6_1 ctx), ((parallel_generatedRoot_197.w_6_2 ctx), ((parallel_generatedRoot_197.w_6_3 ctx), ((parallel_generatedRoot_197.w_6_4 ctx), ((parallel_generatedRoot_197.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_197.w_9_0 ctx), ((parallel_generatedRoot_197.w_12_0 ctx), ((parallel_generatedRoot_197.w_15_0 ctx), ((parallel_generatedRoot_197.w_18_0 ctx), ((parallel_generatedRoot_197.w_21_0 ctx), ((parallel_generatedRoot_197.w_24_0 ctx), ())))))) ((parallel_generatedRoot_197.w_25_0 ctx), ((parallel_generatedRoot_197.w_25_1 ctx), ((parallel_generatedRoot_197.w_25_2 ctx), ((parallel_generatedRoot_197.w_25_3 ctx), ((parallel_generatedRoot_197.w_25_4 ctx), ((parallel_generatedRoot_197.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_197.outs ctx

theorem parallel_generatedRoot_197.facts_0  {ctx : parallel_generatedRoot_197Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_197.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_197.w_0_0 ctx) (parallel_generatedRoot_197.w_1_0 ctx) (parallel_generatedRoot_197.w_2_0 ctx) (parallel_generatedRoot_197.w_3_0 ctx) (parallel_generatedRoot_197.w_4_0 ctx) (parallel_generatedRoot_197.w_5_0 ctx) (parallel_generatedRoot_197.w_6_0 ctx) (parallel_generatedRoot_197.w_6_1 ctx) (parallel_generatedRoot_197.w_6_2 ctx) (parallel_generatedRoot_197.w_6_3 ctx) (parallel_generatedRoot_197.w_6_4 ctx) (parallel_generatedRoot_197.w_6_5 ctx) (parallel_generatedRoot_197.w_7_0 ctx) (parallel_generatedRoot_197.w_8_0 ctx) (parallel_generatedRoot_197.w_9_0 ctx) (parallel_generatedRoot_197.w_10_0 ctx) (parallel_generatedRoot_197.w_11_0 ctx) (parallel_generatedRoot_197.w_12_0 ctx) (parallel_generatedRoot_197.w_13_0 ctx) (parallel_generatedRoot_197.w_14_0 ctx) (parallel_generatedRoot_197.w_15_0 ctx) (parallel_generatedRoot_197.w_16_0 ctx) (parallel_generatedRoot_197.w_17_0 ctx) (parallel_generatedRoot_197.w_18_0 ctx) (parallel_generatedRoot_197.w_19_0 ctx) (parallel_generatedRoot_197.w_20_0 ctx) (parallel_generatedRoot_197.w_21_0 ctx) (parallel_generatedRoot_197.w_22_0 ctx) (parallel_generatedRoot_197.w_23_0 ctx) (parallel_generatedRoot_197.w_24_0 ctx) (parallel_generatedRoot_197.w_25_0 ctx) (parallel_generatedRoot_197.w_25_1 ctx) (parallel_generatedRoot_197.w_25_2 ctx) (parallel_generatedRoot_197.w_25_3 ctx) (parallel_generatedRoot_197.w_25_4 ctx) (parallel_generatedRoot_197.w_25_5 ctx) outputs) :
    parallel_generatedRoot_197.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_197.Facts  (ctx : parallel_generatedRoot_197Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_197.Facts_0 ctx outputs

theorem parallel_generatedRoot_197.facts  (ctx : parallel_generatedRoot_197Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_197.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_197.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_197.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_201.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_201.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_201Ctx" => (parallel_generatedRoot_201.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_201.w_0_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_201.w_1_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_201.w_2_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_201.w_3_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_201.w_4_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_201.w_5_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_201.w_6_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_201.w_6_1  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_201.w_6_2  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_201.w_6_3  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_201.w_6_4  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_201.w_6_5  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_201.w_7_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_201.w_8_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_201.w_9_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_7_0 ctx) (parallel_generatedRoot_201.w_8_0 ctx)

def parallel_generatedRoot_201.w_10_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_201.w_11_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_201.w_12_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_10_0 ctx) (parallel_generatedRoot_201.w_11_0 ctx)

def parallel_generatedRoot_201.w_13_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_201.w_14_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_201.w_15_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_13_0 ctx) (parallel_generatedRoot_201.w_14_0 ctx)

def parallel_generatedRoot_201.w_16_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_201.w_17_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_201.w_18_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_16_0 ctx) (parallel_generatedRoot_201.w_17_0 ctx)

def parallel_generatedRoot_201.w_19_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_201.w_20_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_201.w_21_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_19_0 ctx) (parallel_generatedRoot_201.w_20_0 ctx)

def parallel_generatedRoot_201.w_22_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_201.w_23_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_201.w_24_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_22_0 ctx) (parallel_generatedRoot_201.w_23_0 ctx)

def parallel_generatedRoot_201.w_25_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_201.w_25_1  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_201.w_25_2  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_201.w_25_3  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_201.w_25_4  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_201.w_25_5  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_201.outs  (ctx : parallel_generatedRoot_201Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_201.w_25_0 ctx), ((parallel_generatedRoot_201.w_25_1 ctx), ((parallel_generatedRoot_201.w_25_2 ctx), ((parallel_generatedRoot_201.w_25_3 ctx), ((parallel_generatedRoot_201.w_25_4 ctx), ((parallel_generatedRoot_201.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_201.Facts_0  (ctx : parallel_generatedRoot_201Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_201_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_201.w_0_0 ctx), ((parallel_generatedRoot_201.w_1_0 ctx), ((parallel_generatedRoot_201.w_2_0 ctx), ((parallel_generatedRoot_201.w_3_0 ctx), ((parallel_generatedRoot_201.w_4_0 ctx), ((parallel_generatedRoot_201.w_5_0 ctx), ())))))) (((parallel_generatedRoot_201.w_6_0 ctx) i), (((parallel_generatedRoot_201.w_6_1 ctx) i), (((parallel_generatedRoot_201.w_6_2 ctx) i), (((parallel_generatedRoot_201.w_6_3 ctx) i), (((parallel_generatedRoot_201.w_6_4 ctx) i), (((parallel_generatedRoot_201.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_0 ctx) (0) (parallel_generatedRoot_201.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_0 ctx) (1) (parallel_generatedRoot_201.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_1 ctx) (0) (parallel_generatedRoot_201.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_1 ctx) (1) (parallel_generatedRoot_201.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_2 ctx) (0) (parallel_generatedRoot_201.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_2 ctx) (1) (parallel_generatedRoot_201.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_3 ctx) (0) (parallel_generatedRoot_201.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_3 ctx) (1) (parallel_generatedRoot_201.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_4 ctx) (0) (parallel_generatedRoot_201.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_4 ctx) (1) (parallel_generatedRoot_201.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_5 ctx) (0) (parallel_generatedRoot_201.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_6_5 ctx) (1) (parallel_generatedRoot_201.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_201_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_201.w_6_0 ctx), ((parallel_generatedRoot_201.w_6_1 ctx), ((parallel_generatedRoot_201.w_6_2 ctx), ((parallel_generatedRoot_201.w_6_3 ctx), ((parallel_generatedRoot_201.w_6_4 ctx), ((parallel_generatedRoot_201.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_201.w_9_0 ctx), ((parallel_generatedRoot_201.w_12_0 ctx), ((parallel_generatedRoot_201.w_15_0 ctx), ((parallel_generatedRoot_201.w_18_0 ctx), ((parallel_generatedRoot_201.w_21_0 ctx), ((parallel_generatedRoot_201.w_24_0 ctx), ())))))) ((parallel_generatedRoot_201.w_25_0 ctx), ((parallel_generatedRoot_201.w_25_1 ctx), ((parallel_generatedRoot_201.w_25_2 ctx), ((parallel_generatedRoot_201.w_25_3 ctx), ((parallel_generatedRoot_201.w_25_4 ctx), ((parallel_generatedRoot_201.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_201.outs ctx

theorem parallel_generatedRoot_201.facts_0  {ctx : parallel_generatedRoot_201Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_201.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_201.w_0_0 ctx) (parallel_generatedRoot_201.w_1_0 ctx) (parallel_generatedRoot_201.w_2_0 ctx) (parallel_generatedRoot_201.w_3_0 ctx) (parallel_generatedRoot_201.w_4_0 ctx) (parallel_generatedRoot_201.w_5_0 ctx) (parallel_generatedRoot_201.w_6_0 ctx) (parallel_generatedRoot_201.w_6_1 ctx) (parallel_generatedRoot_201.w_6_2 ctx) (parallel_generatedRoot_201.w_6_3 ctx) (parallel_generatedRoot_201.w_6_4 ctx) (parallel_generatedRoot_201.w_6_5 ctx) (parallel_generatedRoot_201.w_7_0 ctx) (parallel_generatedRoot_201.w_8_0 ctx) (parallel_generatedRoot_201.w_9_0 ctx) (parallel_generatedRoot_201.w_10_0 ctx) (parallel_generatedRoot_201.w_11_0 ctx) (parallel_generatedRoot_201.w_12_0 ctx) (parallel_generatedRoot_201.w_13_0 ctx) (parallel_generatedRoot_201.w_14_0 ctx) (parallel_generatedRoot_201.w_15_0 ctx) (parallel_generatedRoot_201.w_16_0 ctx) (parallel_generatedRoot_201.w_17_0 ctx) (parallel_generatedRoot_201.w_18_0 ctx) (parallel_generatedRoot_201.w_19_0 ctx) (parallel_generatedRoot_201.w_20_0 ctx) (parallel_generatedRoot_201.w_21_0 ctx) (parallel_generatedRoot_201.w_22_0 ctx) (parallel_generatedRoot_201.w_23_0 ctx) (parallel_generatedRoot_201.w_24_0 ctx) (parallel_generatedRoot_201.w_25_0 ctx) (parallel_generatedRoot_201.w_25_1 ctx) (parallel_generatedRoot_201.w_25_2 ctx) (parallel_generatedRoot_201.w_25_3 ctx) (parallel_generatedRoot_201.w_25_4 ctx) (parallel_generatedRoot_201.w_25_5 ctx) outputs) :
    parallel_generatedRoot_201.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_201.Facts  (ctx : parallel_generatedRoot_201Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_201.Facts_0 ctx outputs

theorem parallel_generatedRoot_201.facts  (ctx : parallel_generatedRoot_201Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_201.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_201.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_201.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_205.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_205.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_205Ctx" => (parallel_generatedRoot_205.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_205.w_0_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_205.w_1_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_205.w_2_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_205.w_3_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_205.w_4_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_205.w_5_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_205.w_6_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_205.w_6_1  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_205.w_6_2  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_205.w_6_3  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_205.w_6_4  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_205.w_6_5  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_205.w_7_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_205.w_8_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_205.w_9_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_7_0 ctx) (parallel_generatedRoot_205.w_8_0 ctx)

def parallel_generatedRoot_205.w_10_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_205.w_11_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_205.w_12_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_10_0 ctx) (parallel_generatedRoot_205.w_11_0 ctx)

def parallel_generatedRoot_205.w_13_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_205.w_14_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_205.w_15_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_13_0 ctx) (parallel_generatedRoot_205.w_14_0 ctx)

def parallel_generatedRoot_205.w_16_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_205.w_17_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_205.w_18_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_16_0 ctx) (parallel_generatedRoot_205.w_17_0 ctx)

def parallel_generatedRoot_205.w_19_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_205.w_20_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_205.w_21_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_19_0 ctx) (parallel_generatedRoot_205.w_20_0 ctx)

def parallel_generatedRoot_205.w_22_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_205.w_23_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_205.w_24_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_22_0 ctx) (parallel_generatedRoot_205.w_23_0 ctx)

def parallel_generatedRoot_205.w_25_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_205.w_25_1  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_205.w_25_2  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_205.w_25_3  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_205.w_25_4  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_205.w_25_5  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_205.outs  (ctx : parallel_generatedRoot_205Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_205.w_25_0 ctx), ((parallel_generatedRoot_205.w_25_1 ctx), ((parallel_generatedRoot_205.w_25_2 ctx), ((parallel_generatedRoot_205.w_25_3 ctx), ((parallel_generatedRoot_205.w_25_4 ctx), ((parallel_generatedRoot_205.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_205.Facts_0  (ctx : parallel_generatedRoot_205Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_205_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_205.w_0_0 ctx), ((parallel_generatedRoot_205.w_1_0 ctx), ((parallel_generatedRoot_205.w_2_0 ctx), ((parallel_generatedRoot_205.w_3_0 ctx), ((parallel_generatedRoot_205.w_4_0 ctx), ((parallel_generatedRoot_205.w_5_0 ctx), ())))))) (((parallel_generatedRoot_205.w_6_0 ctx) i), (((parallel_generatedRoot_205.w_6_1 ctx) i), (((parallel_generatedRoot_205.w_6_2 ctx) i), (((parallel_generatedRoot_205.w_6_3 ctx) i), (((parallel_generatedRoot_205.w_6_4 ctx) i), (((parallel_generatedRoot_205.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_0 ctx) (0) (parallel_generatedRoot_205.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_0 ctx) (1) (parallel_generatedRoot_205.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_1 ctx) (0) (parallel_generatedRoot_205.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_1 ctx) (1) (parallel_generatedRoot_205.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_2 ctx) (0) (parallel_generatedRoot_205.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_2 ctx) (1) (parallel_generatedRoot_205.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_3 ctx) (0) (parallel_generatedRoot_205.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_3 ctx) (1) (parallel_generatedRoot_205.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_4 ctx) (0) (parallel_generatedRoot_205.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_4 ctx) (1) (parallel_generatedRoot_205.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_5 ctx) (0) (parallel_generatedRoot_205.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_6_5 ctx) (1) (parallel_generatedRoot_205.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_205_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_205.w_6_0 ctx), ((parallel_generatedRoot_205.w_6_1 ctx), ((parallel_generatedRoot_205.w_6_2 ctx), ((parallel_generatedRoot_205.w_6_3 ctx), ((parallel_generatedRoot_205.w_6_4 ctx), ((parallel_generatedRoot_205.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_205.w_9_0 ctx), ((parallel_generatedRoot_205.w_12_0 ctx), ((parallel_generatedRoot_205.w_15_0 ctx), ((parallel_generatedRoot_205.w_18_0 ctx), ((parallel_generatedRoot_205.w_21_0 ctx), ((parallel_generatedRoot_205.w_24_0 ctx), ())))))) ((parallel_generatedRoot_205.w_25_0 ctx), ((parallel_generatedRoot_205.w_25_1 ctx), ((parallel_generatedRoot_205.w_25_2 ctx), ((parallel_generatedRoot_205.w_25_3 ctx), ((parallel_generatedRoot_205.w_25_4 ctx), ((parallel_generatedRoot_205.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_205.outs ctx

theorem parallel_generatedRoot_205.facts_0  {ctx : parallel_generatedRoot_205Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_205.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_205.w_0_0 ctx) (parallel_generatedRoot_205.w_1_0 ctx) (parallel_generatedRoot_205.w_2_0 ctx) (parallel_generatedRoot_205.w_3_0 ctx) (parallel_generatedRoot_205.w_4_0 ctx) (parallel_generatedRoot_205.w_5_0 ctx) (parallel_generatedRoot_205.w_6_0 ctx) (parallel_generatedRoot_205.w_6_1 ctx) (parallel_generatedRoot_205.w_6_2 ctx) (parallel_generatedRoot_205.w_6_3 ctx) (parallel_generatedRoot_205.w_6_4 ctx) (parallel_generatedRoot_205.w_6_5 ctx) (parallel_generatedRoot_205.w_7_0 ctx) (parallel_generatedRoot_205.w_8_0 ctx) (parallel_generatedRoot_205.w_9_0 ctx) (parallel_generatedRoot_205.w_10_0 ctx) (parallel_generatedRoot_205.w_11_0 ctx) (parallel_generatedRoot_205.w_12_0 ctx) (parallel_generatedRoot_205.w_13_0 ctx) (parallel_generatedRoot_205.w_14_0 ctx) (parallel_generatedRoot_205.w_15_0 ctx) (parallel_generatedRoot_205.w_16_0 ctx) (parallel_generatedRoot_205.w_17_0 ctx) (parallel_generatedRoot_205.w_18_0 ctx) (parallel_generatedRoot_205.w_19_0 ctx) (parallel_generatedRoot_205.w_20_0 ctx) (parallel_generatedRoot_205.w_21_0 ctx) (parallel_generatedRoot_205.w_22_0 ctx) (parallel_generatedRoot_205.w_23_0 ctx) (parallel_generatedRoot_205.w_24_0 ctx) (parallel_generatedRoot_205.w_25_0 ctx) (parallel_generatedRoot_205.w_25_1 ctx) (parallel_generatedRoot_205.w_25_2 ctx) (parallel_generatedRoot_205.w_25_3 ctx) (parallel_generatedRoot_205.w_25_4 ctx) (parallel_generatedRoot_205.w_25_5 ctx) outputs) :
    parallel_generatedRoot_205.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_205.Facts  (ctx : parallel_generatedRoot_205Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_205.Facts_0 ctx outputs

theorem parallel_generatedRoot_205.facts  (ctx : parallel_generatedRoot_205Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_205.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_205.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_205.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_209.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_209.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_209Ctx" => (parallel_generatedRoot_209.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_209.w_0_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_209.w_1_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_209.w_2_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_209.w_3_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_209.w_4_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_209.w_5_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_209.w_6_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_209.w_6_1  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_209.w_6_2  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_209.w_6_3  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_209.w_6_4  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_209.w_6_5  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_209.w_7_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_209.w_8_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_209.w_9_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_7_0 ctx) (parallel_generatedRoot_209.w_8_0 ctx)

def parallel_generatedRoot_209.w_10_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_209.w_11_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_209.w_12_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_10_0 ctx) (parallel_generatedRoot_209.w_11_0 ctx)

def parallel_generatedRoot_209.w_13_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_209.w_14_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_209.w_15_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_13_0 ctx) (parallel_generatedRoot_209.w_14_0 ctx)

def parallel_generatedRoot_209.w_16_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_209.w_17_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_209.w_18_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_16_0 ctx) (parallel_generatedRoot_209.w_17_0 ctx)

def parallel_generatedRoot_209.w_19_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_209.w_20_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_209.w_21_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_19_0 ctx) (parallel_generatedRoot_209.w_20_0 ctx)

def parallel_generatedRoot_209.w_22_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_209.w_23_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_209.w_24_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_22_0 ctx) (parallel_generatedRoot_209.w_23_0 ctx)

def parallel_generatedRoot_209.w_25_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_209.w_25_1  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_209.w_25_2  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_209.w_25_3  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_209.w_25_4  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_209.w_25_5  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_209.outs  (ctx : parallel_generatedRoot_209Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_209.w_25_0 ctx), ((parallel_generatedRoot_209.w_25_1 ctx), ((parallel_generatedRoot_209.w_25_2 ctx), ((parallel_generatedRoot_209.w_25_3 ctx), ((parallel_generatedRoot_209.w_25_4 ctx), ((parallel_generatedRoot_209.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_209.Facts_0  (ctx : parallel_generatedRoot_209Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_209_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_209.w_0_0 ctx), ((parallel_generatedRoot_209.w_1_0 ctx), ((parallel_generatedRoot_209.w_2_0 ctx), ((parallel_generatedRoot_209.w_3_0 ctx), ((parallel_generatedRoot_209.w_4_0 ctx), ((parallel_generatedRoot_209.w_5_0 ctx), ())))))) (((parallel_generatedRoot_209.w_6_0 ctx) i), (((parallel_generatedRoot_209.w_6_1 ctx) i), (((parallel_generatedRoot_209.w_6_2 ctx) i), (((parallel_generatedRoot_209.w_6_3 ctx) i), (((parallel_generatedRoot_209.w_6_4 ctx) i), (((parallel_generatedRoot_209.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_0 ctx) (0) (parallel_generatedRoot_209.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_0 ctx) (1) (parallel_generatedRoot_209.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_1 ctx) (0) (parallel_generatedRoot_209.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_1 ctx) (1) (parallel_generatedRoot_209.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_2 ctx) (0) (parallel_generatedRoot_209.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_2 ctx) (1) (parallel_generatedRoot_209.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_3 ctx) (0) (parallel_generatedRoot_209.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_3 ctx) (1) (parallel_generatedRoot_209.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_4 ctx) (0) (parallel_generatedRoot_209.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_4 ctx) (1) (parallel_generatedRoot_209.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_5 ctx) (0) (parallel_generatedRoot_209.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_6_5 ctx) (1) (parallel_generatedRoot_209.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_209_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_209.w_6_0 ctx), ((parallel_generatedRoot_209.w_6_1 ctx), ((parallel_generatedRoot_209.w_6_2 ctx), ((parallel_generatedRoot_209.w_6_3 ctx), ((parallel_generatedRoot_209.w_6_4 ctx), ((parallel_generatedRoot_209.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_209.w_9_0 ctx), ((parallel_generatedRoot_209.w_12_0 ctx), ((parallel_generatedRoot_209.w_15_0 ctx), ((parallel_generatedRoot_209.w_18_0 ctx), ((parallel_generatedRoot_209.w_21_0 ctx), ((parallel_generatedRoot_209.w_24_0 ctx), ())))))) ((parallel_generatedRoot_209.w_25_0 ctx), ((parallel_generatedRoot_209.w_25_1 ctx), ((parallel_generatedRoot_209.w_25_2 ctx), ((parallel_generatedRoot_209.w_25_3 ctx), ((parallel_generatedRoot_209.w_25_4 ctx), ((parallel_generatedRoot_209.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_209.outs ctx

theorem parallel_generatedRoot_209.facts_0  {ctx : parallel_generatedRoot_209Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_209.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_209.w_0_0 ctx) (parallel_generatedRoot_209.w_1_0 ctx) (parallel_generatedRoot_209.w_2_0 ctx) (parallel_generatedRoot_209.w_3_0 ctx) (parallel_generatedRoot_209.w_4_0 ctx) (parallel_generatedRoot_209.w_5_0 ctx) (parallel_generatedRoot_209.w_6_0 ctx) (parallel_generatedRoot_209.w_6_1 ctx) (parallel_generatedRoot_209.w_6_2 ctx) (parallel_generatedRoot_209.w_6_3 ctx) (parallel_generatedRoot_209.w_6_4 ctx) (parallel_generatedRoot_209.w_6_5 ctx) (parallel_generatedRoot_209.w_7_0 ctx) (parallel_generatedRoot_209.w_8_0 ctx) (parallel_generatedRoot_209.w_9_0 ctx) (parallel_generatedRoot_209.w_10_0 ctx) (parallel_generatedRoot_209.w_11_0 ctx) (parallel_generatedRoot_209.w_12_0 ctx) (parallel_generatedRoot_209.w_13_0 ctx) (parallel_generatedRoot_209.w_14_0 ctx) (parallel_generatedRoot_209.w_15_0 ctx) (parallel_generatedRoot_209.w_16_0 ctx) (parallel_generatedRoot_209.w_17_0 ctx) (parallel_generatedRoot_209.w_18_0 ctx) (parallel_generatedRoot_209.w_19_0 ctx) (parallel_generatedRoot_209.w_20_0 ctx) (parallel_generatedRoot_209.w_21_0 ctx) (parallel_generatedRoot_209.w_22_0 ctx) (parallel_generatedRoot_209.w_23_0 ctx) (parallel_generatedRoot_209.w_24_0 ctx) (parallel_generatedRoot_209.w_25_0 ctx) (parallel_generatedRoot_209.w_25_1 ctx) (parallel_generatedRoot_209.w_25_2 ctx) (parallel_generatedRoot_209.w_25_3 ctx) (parallel_generatedRoot_209.w_25_4 ctx) (parallel_generatedRoot_209.w_25_5 ctx) outputs) :
    parallel_generatedRoot_209.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_209.Facts  (ctx : parallel_generatedRoot_209Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_209.Facts_0 ctx outputs

theorem parallel_generatedRoot_209.facts  (ctx : parallel_generatedRoot_209Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_209.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_209.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_209.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_213.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_213.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_213Ctx" => (parallel_generatedRoot_213.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_213.w_0_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_213.w_1_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_213.w_2_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_213.w_3_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_213.w_4_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_213.w_5_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_213.w_6_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_213.w_6_1  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_213.w_6_2  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_213.w_6_3  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_213.w_6_4  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_213.w_6_5  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_213.w_7_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_213.w_8_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_213.w_9_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_7_0 ctx) (parallel_generatedRoot_213.w_8_0 ctx)

def parallel_generatedRoot_213.w_10_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_213.w_11_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_213.w_12_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_10_0 ctx) (parallel_generatedRoot_213.w_11_0 ctx)

def parallel_generatedRoot_213.w_13_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_213.w_14_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_213.w_15_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_13_0 ctx) (parallel_generatedRoot_213.w_14_0 ctx)

def parallel_generatedRoot_213.w_16_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_213.w_17_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_213.w_18_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_16_0 ctx) (parallel_generatedRoot_213.w_17_0 ctx)

def parallel_generatedRoot_213.w_19_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_213.w_20_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_213.w_21_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_19_0 ctx) (parallel_generatedRoot_213.w_20_0 ctx)

def parallel_generatedRoot_213.w_22_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_213.w_23_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_213.w_24_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_22_0 ctx) (parallel_generatedRoot_213.w_23_0 ctx)

def parallel_generatedRoot_213.w_25_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_213.w_25_1  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_213.w_25_2  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_213.w_25_3  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_213.w_25_4  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_213.w_25_5  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_213.outs  (ctx : parallel_generatedRoot_213Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_213.w_25_0 ctx), ((parallel_generatedRoot_213.w_25_1 ctx), ((parallel_generatedRoot_213.w_25_2 ctx), ((parallel_generatedRoot_213.w_25_3 ctx), ((parallel_generatedRoot_213.w_25_4 ctx), ((parallel_generatedRoot_213.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_213.Facts_0  (ctx : parallel_generatedRoot_213Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_213_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_213.w_0_0 ctx), ((parallel_generatedRoot_213.w_1_0 ctx), ((parallel_generatedRoot_213.w_2_0 ctx), ((parallel_generatedRoot_213.w_3_0 ctx), ((parallel_generatedRoot_213.w_4_0 ctx), ((parallel_generatedRoot_213.w_5_0 ctx), ())))))) (((parallel_generatedRoot_213.w_6_0 ctx) i), (((parallel_generatedRoot_213.w_6_1 ctx) i), (((parallel_generatedRoot_213.w_6_2 ctx) i), (((parallel_generatedRoot_213.w_6_3 ctx) i), (((parallel_generatedRoot_213.w_6_4 ctx) i), (((parallel_generatedRoot_213.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_0 ctx) (0) (parallel_generatedRoot_213.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_0 ctx) (1) (parallel_generatedRoot_213.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_1 ctx) (0) (parallel_generatedRoot_213.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_1 ctx) (1) (parallel_generatedRoot_213.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_2 ctx) (0) (parallel_generatedRoot_213.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_2 ctx) (1) (parallel_generatedRoot_213.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_3 ctx) (0) (parallel_generatedRoot_213.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_3 ctx) (1) (parallel_generatedRoot_213.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_4 ctx) (0) (parallel_generatedRoot_213.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_4 ctx) (1) (parallel_generatedRoot_213.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_5 ctx) (0) (parallel_generatedRoot_213.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_6_5 ctx) (1) (parallel_generatedRoot_213.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_213_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_213.w_6_0 ctx), ((parallel_generatedRoot_213.w_6_1 ctx), ((parallel_generatedRoot_213.w_6_2 ctx), ((parallel_generatedRoot_213.w_6_3 ctx), ((parallel_generatedRoot_213.w_6_4 ctx), ((parallel_generatedRoot_213.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_213.w_9_0 ctx), ((parallel_generatedRoot_213.w_12_0 ctx), ((parallel_generatedRoot_213.w_15_0 ctx), ((parallel_generatedRoot_213.w_18_0 ctx), ((parallel_generatedRoot_213.w_21_0 ctx), ((parallel_generatedRoot_213.w_24_0 ctx), ())))))) ((parallel_generatedRoot_213.w_25_0 ctx), ((parallel_generatedRoot_213.w_25_1 ctx), ((parallel_generatedRoot_213.w_25_2 ctx), ((parallel_generatedRoot_213.w_25_3 ctx), ((parallel_generatedRoot_213.w_25_4 ctx), ((parallel_generatedRoot_213.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_213.outs ctx

theorem parallel_generatedRoot_213.facts_0  {ctx : parallel_generatedRoot_213Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_213.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_213.w_0_0 ctx) (parallel_generatedRoot_213.w_1_0 ctx) (parallel_generatedRoot_213.w_2_0 ctx) (parallel_generatedRoot_213.w_3_0 ctx) (parallel_generatedRoot_213.w_4_0 ctx) (parallel_generatedRoot_213.w_5_0 ctx) (parallel_generatedRoot_213.w_6_0 ctx) (parallel_generatedRoot_213.w_6_1 ctx) (parallel_generatedRoot_213.w_6_2 ctx) (parallel_generatedRoot_213.w_6_3 ctx) (parallel_generatedRoot_213.w_6_4 ctx) (parallel_generatedRoot_213.w_6_5 ctx) (parallel_generatedRoot_213.w_7_0 ctx) (parallel_generatedRoot_213.w_8_0 ctx) (parallel_generatedRoot_213.w_9_0 ctx) (parallel_generatedRoot_213.w_10_0 ctx) (parallel_generatedRoot_213.w_11_0 ctx) (parallel_generatedRoot_213.w_12_0 ctx) (parallel_generatedRoot_213.w_13_0 ctx) (parallel_generatedRoot_213.w_14_0 ctx) (parallel_generatedRoot_213.w_15_0 ctx) (parallel_generatedRoot_213.w_16_0 ctx) (parallel_generatedRoot_213.w_17_0 ctx) (parallel_generatedRoot_213.w_18_0 ctx) (parallel_generatedRoot_213.w_19_0 ctx) (parallel_generatedRoot_213.w_20_0 ctx) (parallel_generatedRoot_213.w_21_0 ctx) (parallel_generatedRoot_213.w_22_0 ctx) (parallel_generatedRoot_213.w_23_0 ctx) (parallel_generatedRoot_213.w_24_0 ctx) (parallel_generatedRoot_213.w_25_0 ctx) (parallel_generatedRoot_213.w_25_1 ctx) (parallel_generatedRoot_213.w_25_2 ctx) (parallel_generatedRoot_213.w_25_3 ctx) (parallel_generatedRoot_213.w_25_4 ctx) (parallel_generatedRoot_213.w_25_5 ctx) outputs) :
    parallel_generatedRoot_213.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_213.Facts  (ctx : parallel_generatedRoot_213Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_213.Facts_0 ctx outputs

theorem parallel_generatedRoot_213.facts  (ctx : parallel_generatedRoot_213Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_213.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_213.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_213.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_217.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_217.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_217Ctx" => (parallel_generatedRoot_217.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_217.w_0_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_217.w_1_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_217.w_2_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_217.w_3_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_217.w_4_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_217.w_5_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_217.w_6_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_217.w_6_1  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_217.w_6_2  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_217.w_6_3  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_217.w_6_4  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_217.w_6_5  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_217.w_7_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_217.w_8_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_217.w_9_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_7_0 ctx) (parallel_generatedRoot_217.w_8_0 ctx)

def parallel_generatedRoot_217.w_10_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_217.w_11_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_217.w_12_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_10_0 ctx) (parallel_generatedRoot_217.w_11_0 ctx)

def parallel_generatedRoot_217.w_13_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_217.w_14_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_217.w_15_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_13_0 ctx) (parallel_generatedRoot_217.w_14_0 ctx)

def parallel_generatedRoot_217.w_16_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_217.w_17_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_217.w_18_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_16_0 ctx) (parallel_generatedRoot_217.w_17_0 ctx)

def parallel_generatedRoot_217.w_19_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_217.w_20_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_217.w_21_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_19_0 ctx) (parallel_generatedRoot_217.w_20_0 ctx)

def parallel_generatedRoot_217.w_22_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_217.w_23_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_217.w_24_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_22_0 ctx) (parallel_generatedRoot_217.w_23_0 ctx)

def parallel_generatedRoot_217.w_25_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_217.w_25_1  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_217.w_25_2  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_217.w_25_3  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_217.w_25_4  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_217.w_25_5  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_217.outs  (ctx : parallel_generatedRoot_217Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_217.w_25_0 ctx), ((parallel_generatedRoot_217.w_25_1 ctx), ((parallel_generatedRoot_217.w_25_2 ctx), ((parallel_generatedRoot_217.w_25_3 ctx), ((parallel_generatedRoot_217.w_25_4 ctx), ((parallel_generatedRoot_217.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_217.Facts_0  (ctx : parallel_generatedRoot_217Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_217_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_217.w_0_0 ctx), ((parallel_generatedRoot_217.w_1_0 ctx), ((parallel_generatedRoot_217.w_2_0 ctx), ((parallel_generatedRoot_217.w_3_0 ctx), ((parallel_generatedRoot_217.w_4_0 ctx), ((parallel_generatedRoot_217.w_5_0 ctx), ())))))) (((parallel_generatedRoot_217.w_6_0 ctx) i), (((parallel_generatedRoot_217.w_6_1 ctx) i), (((parallel_generatedRoot_217.w_6_2 ctx) i), (((parallel_generatedRoot_217.w_6_3 ctx) i), (((parallel_generatedRoot_217.w_6_4 ctx) i), (((parallel_generatedRoot_217.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_0 ctx) (0) (parallel_generatedRoot_217.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_0 ctx) (1) (parallel_generatedRoot_217.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_1 ctx) (0) (parallel_generatedRoot_217.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_1 ctx) (1) (parallel_generatedRoot_217.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_2 ctx) (0) (parallel_generatedRoot_217.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_2 ctx) (1) (parallel_generatedRoot_217.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_3 ctx) (0) (parallel_generatedRoot_217.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_3 ctx) (1) (parallel_generatedRoot_217.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_4 ctx) (0) (parallel_generatedRoot_217.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_4 ctx) (1) (parallel_generatedRoot_217.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_5 ctx) (0) (parallel_generatedRoot_217.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_6_5 ctx) (1) (parallel_generatedRoot_217.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_217_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_217.w_6_0 ctx), ((parallel_generatedRoot_217.w_6_1 ctx), ((parallel_generatedRoot_217.w_6_2 ctx), ((parallel_generatedRoot_217.w_6_3 ctx), ((parallel_generatedRoot_217.w_6_4 ctx), ((parallel_generatedRoot_217.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_217.w_9_0 ctx), ((parallel_generatedRoot_217.w_12_0 ctx), ((parallel_generatedRoot_217.w_15_0 ctx), ((parallel_generatedRoot_217.w_18_0 ctx), ((parallel_generatedRoot_217.w_21_0 ctx), ((parallel_generatedRoot_217.w_24_0 ctx), ())))))) ((parallel_generatedRoot_217.w_25_0 ctx), ((parallel_generatedRoot_217.w_25_1 ctx), ((parallel_generatedRoot_217.w_25_2 ctx), ((parallel_generatedRoot_217.w_25_3 ctx), ((parallel_generatedRoot_217.w_25_4 ctx), ((parallel_generatedRoot_217.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_217.outs ctx

theorem parallel_generatedRoot_217.facts_0  {ctx : parallel_generatedRoot_217Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_217.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_217.w_0_0 ctx) (parallel_generatedRoot_217.w_1_0 ctx) (parallel_generatedRoot_217.w_2_0 ctx) (parallel_generatedRoot_217.w_3_0 ctx) (parallel_generatedRoot_217.w_4_0 ctx) (parallel_generatedRoot_217.w_5_0 ctx) (parallel_generatedRoot_217.w_6_0 ctx) (parallel_generatedRoot_217.w_6_1 ctx) (parallel_generatedRoot_217.w_6_2 ctx) (parallel_generatedRoot_217.w_6_3 ctx) (parallel_generatedRoot_217.w_6_4 ctx) (parallel_generatedRoot_217.w_6_5 ctx) (parallel_generatedRoot_217.w_7_0 ctx) (parallel_generatedRoot_217.w_8_0 ctx) (parallel_generatedRoot_217.w_9_0 ctx) (parallel_generatedRoot_217.w_10_0 ctx) (parallel_generatedRoot_217.w_11_0 ctx) (parallel_generatedRoot_217.w_12_0 ctx) (parallel_generatedRoot_217.w_13_0 ctx) (parallel_generatedRoot_217.w_14_0 ctx) (parallel_generatedRoot_217.w_15_0 ctx) (parallel_generatedRoot_217.w_16_0 ctx) (parallel_generatedRoot_217.w_17_0 ctx) (parallel_generatedRoot_217.w_18_0 ctx) (parallel_generatedRoot_217.w_19_0 ctx) (parallel_generatedRoot_217.w_20_0 ctx) (parallel_generatedRoot_217.w_21_0 ctx) (parallel_generatedRoot_217.w_22_0 ctx) (parallel_generatedRoot_217.w_23_0 ctx) (parallel_generatedRoot_217.w_24_0 ctx) (parallel_generatedRoot_217.w_25_0 ctx) (parallel_generatedRoot_217.w_25_1 ctx) (parallel_generatedRoot_217.w_25_2 ctx) (parallel_generatedRoot_217.w_25_3 ctx) (parallel_generatedRoot_217.w_25_4 ctx) (parallel_generatedRoot_217.w_25_5 ctx) outputs) :
    parallel_generatedRoot_217.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_217.Facts  (ctx : parallel_generatedRoot_217Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_217.Facts_0 ctx outputs

theorem parallel_generatedRoot_217.facts  (ctx : parallel_generatedRoot_217Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_217.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_217.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_217.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_221.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_221.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_221Ctx" => (parallel_generatedRoot_221.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_221.w_0_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_221.w_1_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_221.w_2_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_221.w_3_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_221.w_4_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_221.w_5_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_221.w_6_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_221.w_6_1  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_221.w_6_2  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_221.w_6_3  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_221.w_6_4  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_221.w_6_5  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_221.w_7_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_221.w_8_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_221.w_9_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_7_0 ctx) (parallel_generatedRoot_221.w_8_0 ctx)

def parallel_generatedRoot_221.w_10_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_221.w_11_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_221.w_12_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_10_0 ctx) (parallel_generatedRoot_221.w_11_0 ctx)

def parallel_generatedRoot_221.w_13_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_221.w_14_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_221.w_15_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_13_0 ctx) (parallel_generatedRoot_221.w_14_0 ctx)

def parallel_generatedRoot_221.w_16_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_221.w_17_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_221.w_18_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_16_0 ctx) (parallel_generatedRoot_221.w_17_0 ctx)

def parallel_generatedRoot_221.w_19_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_221.w_20_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_221.w_21_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_19_0 ctx) (parallel_generatedRoot_221.w_20_0 ctx)

def parallel_generatedRoot_221.w_22_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_221.w_23_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_221.w_24_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_22_0 ctx) (parallel_generatedRoot_221.w_23_0 ctx)

def parallel_generatedRoot_221.w_25_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_221.w_25_1  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_221.w_25_2  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_221.w_25_3  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_221.w_25_4  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_221.w_25_5  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_221.outs  (ctx : parallel_generatedRoot_221Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_221.w_25_0 ctx), ((parallel_generatedRoot_221.w_25_1 ctx), ((parallel_generatedRoot_221.w_25_2 ctx), ((parallel_generatedRoot_221.w_25_3 ctx), ((parallel_generatedRoot_221.w_25_4 ctx), ((parallel_generatedRoot_221.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_221.Facts_0  (ctx : parallel_generatedRoot_221Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_221_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_221.w_0_0 ctx), ((parallel_generatedRoot_221.w_1_0 ctx), ((parallel_generatedRoot_221.w_2_0 ctx), ((parallel_generatedRoot_221.w_3_0 ctx), ((parallel_generatedRoot_221.w_4_0 ctx), ((parallel_generatedRoot_221.w_5_0 ctx), ())))))) (((parallel_generatedRoot_221.w_6_0 ctx) i), (((parallel_generatedRoot_221.w_6_1 ctx) i), (((parallel_generatedRoot_221.w_6_2 ctx) i), (((parallel_generatedRoot_221.w_6_3 ctx) i), (((parallel_generatedRoot_221.w_6_4 ctx) i), (((parallel_generatedRoot_221.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_0 ctx) (0) (parallel_generatedRoot_221.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_0 ctx) (1) (parallel_generatedRoot_221.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_1 ctx) (0) (parallel_generatedRoot_221.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_1 ctx) (1) (parallel_generatedRoot_221.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_2 ctx) (0) (parallel_generatedRoot_221.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_2 ctx) (1) (parallel_generatedRoot_221.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_3 ctx) (0) (parallel_generatedRoot_221.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_3 ctx) (1) (parallel_generatedRoot_221.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_4 ctx) (0) (parallel_generatedRoot_221.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_4 ctx) (1) (parallel_generatedRoot_221.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_5 ctx) (0) (parallel_generatedRoot_221.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_6_5 ctx) (1) (parallel_generatedRoot_221.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_221_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_221.w_6_0 ctx), ((parallel_generatedRoot_221.w_6_1 ctx), ((parallel_generatedRoot_221.w_6_2 ctx), ((parallel_generatedRoot_221.w_6_3 ctx), ((parallel_generatedRoot_221.w_6_4 ctx), ((parallel_generatedRoot_221.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_221.w_9_0 ctx), ((parallel_generatedRoot_221.w_12_0 ctx), ((parallel_generatedRoot_221.w_15_0 ctx), ((parallel_generatedRoot_221.w_18_0 ctx), ((parallel_generatedRoot_221.w_21_0 ctx), ((parallel_generatedRoot_221.w_24_0 ctx), ())))))) ((parallel_generatedRoot_221.w_25_0 ctx), ((parallel_generatedRoot_221.w_25_1 ctx), ((parallel_generatedRoot_221.w_25_2 ctx), ((parallel_generatedRoot_221.w_25_3 ctx), ((parallel_generatedRoot_221.w_25_4 ctx), ((parallel_generatedRoot_221.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_221.outs ctx

theorem parallel_generatedRoot_221.facts_0  {ctx : parallel_generatedRoot_221Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_221.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_221.w_0_0 ctx) (parallel_generatedRoot_221.w_1_0 ctx) (parallel_generatedRoot_221.w_2_0 ctx) (parallel_generatedRoot_221.w_3_0 ctx) (parallel_generatedRoot_221.w_4_0 ctx) (parallel_generatedRoot_221.w_5_0 ctx) (parallel_generatedRoot_221.w_6_0 ctx) (parallel_generatedRoot_221.w_6_1 ctx) (parallel_generatedRoot_221.w_6_2 ctx) (parallel_generatedRoot_221.w_6_3 ctx) (parallel_generatedRoot_221.w_6_4 ctx) (parallel_generatedRoot_221.w_6_5 ctx) (parallel_generatedRoot_221.w_7_0 ctx) (parallel_generatedRoot_221.w_8_0 ctx) (parallel_generatedRoot_221.w_9_0 ctx) (parallel_generatedRoot_221.w_10_0 ctx) (parallel_generatedRoot_221.w_11_0 ctx) (parallel_generatedRoot_221.w_12_0 ctx) (parallel_generatedRoot_221.w_13_0 ctx) (parallel_generatedRoot_221.w_14_0 ctx) (parallel_generatedRoot_221.w_15_0 ctx) (parallel_generatedRoot_221.w_16_0 ctx) (parallel_generatedRoot_221.w_17_0 ctx) (parallel_generatedRoot_221.w_18_0 ctx) (parallel_generatedRoot_221.w_19_0 ctx) (parallel_generatedRoot_221.w_20_0 ctx) (parallel_generatedRoot_221.w_21_0 ctx) (parallel_generatedRoot_221.w_22_0 ctx) (parallel_generatedRoot_221.w_23_0 ctx) (parallel_generatedRoot_221.w_24_0 ctx) (parallel_generatedRoot_221.w_25_0 ctx) (parallel_generatedRoot_221.w_25_1 ctx) (parallel_generatedRoot_221.w_25_2 ctx) (parallel_generatedRoot_221.w_25_3 ctx) (parallel_generatedRoot_221.w_25_4 ctx) (parallel_generatedRoot_221.w_25_5 ctx) outputs) :
    parallel_generatedRoot_221.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_221.Facts  (ctx : parallel_generatedRoot_221Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_221.Facts_0 ctx outputs

theorem parallel_generatedRoot_221.facts  (ctx : parallel_generatedRoot_221Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_221.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_221.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_221.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_225.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_225.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_225Ctx" => (parallel_generatedRoot_225.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_225.w_0_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_225.w_1_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_225.w_2_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_225.w_3_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_225.w_4_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_225.w_5_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_225.w_6_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_225.w_6_1  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_225.w_6_2  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_225.w_6_3  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_225.w_6_4  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_225.w_6_5  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_225.w_7_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_225.w_8_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_225.w_9_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_7_0 ctx) (parallel_generatedRoot_225.w_8_0 ctx)

def parallel_generatedRoot_225.w_10_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_225.w_11_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_225.w_12_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_10_0 ctx) (parallel_generatedRoot_225.w_11_0 ctx)

def parallel_generatedRoot_225.w_13_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_225.w_14_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_225.w_15_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_13_0 ctx) (parallel_generatedRoot_225.w_14_0 ctx)

def parallel_generatedRoot_225.w_16_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_225.w_17_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_225.w_18_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_16_0 ctx) (parallel_generatedRoot_225.w_17_0 ctx)

def parallel_generatedRoot_225.w_19_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_225.w_20_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_225.w_21_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_19_0 ctx) (parallel_generatedRoot_225.w_20_0 ctx)

def parallel_generatedRoot_225.w_22_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_225.w_23_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_225.w_24_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_22_0 ctx) (parallel_generatedRoot_225.w_23_0 ctx)

def parallel_generatedRoot_225.w_25_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_225.w_25_1  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_225.w_25_2  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_225.w_25_3  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_225.w_25_4  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_225.w_25_5  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_225.outs  (ctx : parallel_generatedRoot_225Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_225.w_25_0 ctx), ((parallel_generatedRoot_225.w_25_1 ctx), ((parallel_generatedRoot_225.w_25_2 ctx), ((parallel_generatedRoot_225.w_25_3 ctx), ((parallel_generatedRoot_225.w_25_4 ctx), ((parallel_generatedRoot_225.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_225.Facts_0  (ctx : parallel_generatedRoot_225Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_225_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_225.w_0_0 ctx), ((parallel_generatedRoot_225.w_1_0 ctx), ((parallel_generatedRoot_225.w_2_0 ctx), ((parallel_generatedRoot_225.w_3_0 ctx), ((parallel_generatedRoot_225.w_4_0 ctx), ((parallel_generatedRoot_225.w_5_0 ctx), ())))))) (((parallel_generatedRoot_225.w_6_0 ctx) i), (((parallel_generatedRoot_225.w_6_1 ctx) i), (((parallel_generatedRoot_225.w_6_2 ctx) i), (((parallel_generatedRoot_225.w_6_3 ctx) i), (((parallel_generatedRoot_225.w_6_4 ctx) i), (((parallel_generatedRoot_225.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_0 ctx) (0) (parallel_generatedRoot_225.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_0 ctx) (1) (parallel_generatedRoot_225.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_1 ctx) (0) (parallel_generatedRoot_225.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_1 ctx) (1) (parallel_generatedRoot_225.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_2 ctx) (0) (parallel_generatedRoot_225.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_2 ctx) (1) (parallel_generatedRoot_225.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_3 ctx) (0) (parallel_generatedRoot_225.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_3 ctx) (1) (parallel_generatedRoot_225.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_4 ctx) (0) (parallel_generatedRoot_225.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_4 ctx) (1) (parallel_generatedRoot_225.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_5 ctx) (0) (parallel_generatedRoot_225.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_6_5 ctx) (1) (parallel_generatedRoot_225.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_225_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_225.w_6_0 ctx), ((parallel_generatedRoot_225.w_6_1 ctx), ((parallel_generatedRoot_225.w_6_2 ctx), ((parallel_generatedRoot_225.w_6_3 ctx), ((parallel_generatedRoot_225.w_6_4 ctx), ((parallel_generatedRoot_225.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_225.w_9_0 ctx), ((parallel_generatedRoot_225.w_12_0 ctx), ((parallel_generatedRoot_225.w_15_0 ctx), ((parallel_generatedRoot_225.w_18_0 ctx), ((parallel_generatedRoot_225.w_21_0 ctx), ((parallel_generatedRoot_225.w_24_0 ctx), ())))))) ((parallel_generatedRoot_225.w_25_0 ctx), ((parallel_generatedRoot_225.w_25_1 ctx), ((parallel_generatedRoot_225.w_25_2 ctx), ((parallel_generatedRoot_225.w_25_3 ctx), ((parallel_generatedRoot_225.w_25_4 ctx), ((parallel_generatedRoot_225.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_225.outs ctx

theorem parallel_generatedRoot_225.facts_0  {ctx : parallel_generatedRoot_225Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_225.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_225.w_0_0 ctx) (parallel_generatedRoot_225.w_1_0 ctx) (parallel_generatedRoot_225.w_2_0 ctx) (parallel_generatedRoot_225.w_3_0 ctx) (parallel_generatedRoot_225.w_4_0 ctx) (parallel_generatedRoot_225.w_5_0 ctx) (parallel_generatedRoot_225.w_6_0 ctx) (parallel_generatedRoot_225.w_6_1 ctx) (parallel_generatedRoot_225.w_6_2 ctx) (parallel_generatedRoot_225.w_6_3 ctx) (parallel_generatedRoot_225.w_6_4 ctx) (parallel_generatedRoot_225.w_6_5 ctx) (parallel_generatedRoot_225.w_7_0 ctx) (parallel_generatedRoot_225.w_8_0 ctx) (parallel_generatedRoot_225.w_9_0 ctx) (parallel_generatedRoot_225.w_10_0 ctx) (parallel_generatedRoot_225.w_11_0 ctx) (parallel_generatedRoot_225.w_12_0 ctx) (parallel_generatedRoot_225.w_13_0 ctx) (parallel_generatedRoot_225.w_14_0 ctx) (parallel_generatedRoot_225.w_15_0 ctx) (parallel_generatedRoot_225.w_16_0 ctx) (parallel_generatedRoot_225.w_17_0 ctx) (parallel_generatedRoot_225.w_18_0 ctx) (parallel_generatedRoot_225.w_19_0 ctx) (parallel_generatedRoot_225.w_20_0 ctx) (parallel_generatedRoot_225.w_21_0 ctx) (parallel_generatedRoot_225.w_22_0 ctx) (parallel_generatedRoot_225.w_23_0 ctx) (parallel_generatedRoot_225.w_24_0 ctx) (parallel_generatedRoot_225.w_25_0 ctx) (parallel_generatedRoot_225.w_25_1 ctx) (parallel_generatedRoot_225.w_25_2 ctx) (parallel_generatedRoot_225.w_25_3 ctx) (parallel_generatedRoot_225.w_25_4 ctx) (parallel_generatedRoot_225.w_25_5 ctx) outputs) :
    parallel_generatedRoot_225.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_225.Facts  (ctx : parallel_generatedRoot_225Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_225.Facts_0 ctx outputs

theorem parallel_generatedRoot_225.facts  (ctx : parallel_generatedRoot_225Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_225.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_225.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_225.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_229.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_229.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_229Ctx" => (parallel_generatedRoot_229.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_229.w_0_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_229.w_1_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_229.w_2_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_229.w_3_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_229.w_4_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_229.w_5_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_229.w_6_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_229.w_6_1  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_229.w_6_2  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_229.w_6_3  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_229.w_6_4  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_229.w_6_5  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_229.w_7_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_229.w_8_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_229.w_9_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_7_0 ctx) (parallel_generatedRoot_229.w_8_0 ctx)

def parallel_generatedRoot_229.w_10_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_229.w_11_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_229.w_12_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_10_0 ctx) (parallel_generatedRoot_229.w_11_0 ctx)

def parallel_generatedRoot_229.w_13_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_229.w_14_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_229.w_15_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_13_0 ctx) (parallel_generatedRoot_229.w_14_0 ctx)

def parallel_generatedRoot_229.w_16_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_229.w_17_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_229.w_18_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_16_0 ctx) (parallel_generatedRoot_229.w_17_0 ctx)

def parallel_generatedRoot_229.w_19_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_229.w_20_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_229.w_21_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_19_0 ctx) (parallel_generatedRoot_229.w_20_0 ctx)

def parallel_generatedRoot_229.w_22_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_229.w_23_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_229.w_24_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_22_0 ctx) (parallel_generatedRoot_229.w_23_0 ctx)

def parallel_generatedRoot_229.w_25_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_229.w_25_1  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_229.w_25_2  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_229.w_25_3  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_229.w_25_4  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_229.w_25_5  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_229.outs  (ctx : parallel_generatedRoot_229Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_229.w_25_0 ctx), ((parallel_generatedRoot_229.w_25_1 ctx), ((parallel_generatedRoot_229.w_25_2 ctx), ((parallel_generatedRoot_229.w_25_3 ctx), ((parallel_generatedRoot_229.w_25_4 ctx), ((parallel_generatedRoot_229.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_229.Facts_0  (ctx : parallel_generatedRoot_229Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_229_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_229.w_0_0 ctx), ((parallel_generatedRoot_229.w_1_0 ctx), ((parallel_generatedRoot_229.w_2_0 ctx), ((parallel_generatedRoot_229.w_3_0 ctx), ((parallel_generatedRoot_229.w_4_0 ctx), ((parallel_generatedRoot_229.w_5_0 ctx), ())))))) (((parallel_generatedRoot_229.w_6_0 ctx) i), (((parallel_generatedRoot_229.w_6_1 ctx) i), (((parallel_generatedRoot_229.w_6_2 ctx) i), (((parallel_generatedRoot_229.w_6_3 ctx) i), (((parallel_generatedRoot_229.w_6_4 ctx) i), (((parallel_generatedRoot_229.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_0 ctx) (0) (parallel_generatedRoot_229.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_0 ctx) (1) (parallel_generatedRoot_229.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_1 ctx) (0) (parallel_generatedRoot_229.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_1 ctx) (1) (parallel_generatedRoot_229.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_2 ctx) (0) (parallel_generatedRoot_229.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_2 ctx) (1) (parallel_generatedRoot_229.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_3 ctx) (0) (parallel_generatedRoot_229.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_3 ctx) (1) (parallel_generatedRoot_229.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_4 ctx) (0) (parallel_generatedRoot_229.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_4 ctx) (1) (parallel_generatedRoot_229.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_5 ctx) (0) (parallel_generatedRoot_229.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_6_5 ctx) (1) (parallel_generatedRoot_229.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_229_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_229.w_6_0 ctx), ((parallel_generatedRoot_229.w_6_1 ctx), ((parallel_generatedRoot_229.w_6_2 ctx), ((parallel_generatedRoot_229.w_6_3 ctx), ((parallel_generatedRoot_229.w_6_4 ctx), ((parallel_generatedRoot_229.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_229.w_9_0 ctx), ((parallel_generatedRoot_229.w_12_0 ctx), ((parallel_generatedRoot_229.w_15_0 ctx), ((parallel_generatedRoot_229.w_18_0 ctx), ((parallel_generatedRoot_229.w_21_0 ctx), ((parallel_generatedRoot_229.w_24_0 ctx), ())))))) ((parallel_generatedRoot_229.w_25_0 ctx), ((parallel_generatedRoot_229.w_25_1 ctx), ((parallel_generatedRoot_229.w_25_2 ctx), ((parallel_generatedRoot_229.w_25_3 ctx), ((parallel_generatedRoot_229.w_25_4 ctx), ((parallel_generatedRoot_229.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_229.outs ctx

theorem parallel_generatedRoot_229.facts_0  {ctx : parallel_generatedRoot_229Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_229.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_229.w_0_0 ctx) (parallel_generatedRoot_229.w_1_0 ctx) (parallel_generatedRoot_229.w_2_0 ctx) (parallel_generatedRoot_229.w_3_0 ctx) (parallel_generatedRoot_229.w_4_0 ctx) (parallel_generatedRoot_229.w_5_0 ctx) (parallel_generatedRoot_229.w_6_0 ctx) (parallel_generatedRoot_229.w_6_1 ctx) (parallel_generatedRoot_229.w_6_2 ctx) (parallel_generatedRoot_229.w_6_3 ctx) (parallel_generatedRoot_229.w_6_4 ctx) (parallel_generatedRoot_229.w_6_5 ctx) (parallel_generatedRoot_229.w_7_0 ctx) (parallel_generatedRoot_229.w_8_0 ctx) (parallel_generatedRoot_229.w_9_0 ctx) (parallel_generatedRoot_229.w_10_0 ctx) (parallel_generatedRoot_229.w_11_0 ctx) (parallel_generatedRoot_229.w_12_0 ctx) (parallel_generatedRoot_229.w_13_0 ctx) (parallel_generatedRoot_229.w_14_0 ctx) (parallel_generatedRoot_229.w_15_0 ctx) (parallel_generatedRoot_229.w_16_0 ctx) (parallel_generatedRoot_229.w_17_0 ctx) (parallel_generatedRoot_229.w_18_0 ctx) (parallel_generatedRoot_229.w_19_0 ctx) (parallel_generatedRoot_229.w_20_0 ctx) (parallel_generatedRoot_229.w_21_0 ctx) (parallel_generatedRoot_229.w_22_0 ctx) (parallel_generatedRoot_229.w_23_0 ctx) (parallel_generatedRoot_229.w_24_0 ctx) (parallel_generatedRoot_229.w_25_0 ctx) (parallel_generatedRoot_229.w_25_1 ctx) (parallel_generatedRoot_229.w_25_2 ctx) (parallel_generatedRoot_229.w_25_3 ctx) (parallel_generatedRoot_229.w_25_4 ctx) (parallel_generatedRoot_229.w_25_5 ctx) outputs) :
    parallel_generatedRoot_229.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_229.Facts  (ctx : parallel_generatedRoot_229Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_229.Facts_0 ctx outputs

theorem parallel_generatedRoot_229.facts  (ctx : parallel_generatedRoot_229Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_229.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_229.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_229.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_233.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_233.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_233Ctx" => (parallel_generatedRoot_233.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_233.w_0_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_233.w_1_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_233.w_2_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_233.w_3_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_233.w_4_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_233.w_5_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_233.w_6_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_233.w_6_1  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_233.w_6_2  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_233.w_6_3  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_233.w_6_4  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_233.w_6_5  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_233.w_7_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_233.w_8_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_233.w_9_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_7_0 ctx) (parallel_generatedRoot_233.w_8_0 ctx)

def parallel_generatedRoot_233.w_10_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_233.w_11_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_233.w_12_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_10_0 ctx) (parallel_generatedRoot_233.w_11_0 ctx)

def parallel_generatedRoot_233.w_13_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_233.w_14_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_233.w_15_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_13_0 ctx) (parallel_generatedRoot_233.w_14_0 ctx)

def parallel_generatedRoot_233.w_16_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_233.w_17_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_233.w_18_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_16_0 ctx) (parallel_generatedRoot_233.w_17_0 ctx)

def parallel_generatedRoot_233.w_19_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_233.w_20_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_233.w_21_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_19_0 ctx) (parallel_generatedRoot_233.w_20_0 ctx)

def parallel_generatedRoot_233.w_22_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_233.w_23_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_233.w_24_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_22_0 ctx) (parallel_generatedRoot_233.w_23_0 ctx)

def parallel_generatedRoot_233.w_25_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_233.w_25_1  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_233.w_25_2  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_233.w_25_3  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_233.w_25_4  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_233.w_25_5  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_233.outs  (ctx : parallel_generatedRoot_233Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_233.w_25_0 ctx), ((parallel_generatedRoot_233.w_25_1 ctx), ((parallel_generatedRoot_233.w_25_2 ctx), ((parallel_generatedRoot_233.w_25_3 ctx), ((parallel_generatedRoot_233.w_25_4 ctx), ((parallel_generatedRoot_233.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_233.Facts_0  (ctx : parallel_generatedRoot_233Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_233_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_233.w_0_0 ctx), ((parallel_generatedRoot_233.w_1_0 ctx), ((parallel_generatedRoot_233.w_2_0 ctx), ((parallel_generatedRoot_233.w_3_0 ctx), ((parallel_generatedRoot_233.w_4_0 ctx), ((parallel_generatedRoot_233.w_5_0 ctx), ())))))) (((parallel_generatedRoot_233.w_6_0 ctx) i), (((parallel_generatedRoot_233.w_6_1 ctx) i), (((parallel_generatedRoot_233.w_6_2 ctx) i), (((parallel_generatedRoot_233.w_6_3 ctx) i), (((parallel_generatedRoot_233.w_6_4 ctx) i), (((parallel_generatedRoot_233.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_0 ctx) (0) (parallel_generatedRoot_233.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_0 ctx) (1) (parallel_generatedRoot_233.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_1 ctx) (0) (parallel_generatedRoot_233.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_1 ctx) (1) (parallel_generatedRoot_233.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_2 ctx) (0) (parallel_generatedRoot_233.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_2 ctx) (1) (parallel_generatedRoot_233.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_3 ctx) (0) (parallel_generatedRoot_233.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_3 ctx) (1) (parallel_generatedRoot_233.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_4 ctx) (0) (parallel_generatedRoot_233.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_4 ctx) (1) (parallel_generatedRoot_233.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_5 ctx) (0) (parallel_generatedRoot_233.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_6_5 ctx) (1) (parallel_generatedRoot_233.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_233_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_233.w_6_0 ctx), ((parallel_generatedRoot_233.w_6_1 ctx), ((parallel_generatedRoot_233.w_6_2 ctx), ((parallel_generatedRoot_233.w_6_3 ctx), ((parallel_generatedRoot_233.w_6_4 ctx), ((parallel_generatedRoot_233.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_233.w_9_0 ctx), ((parallel_generatedRoot_233.w_12_0 ctx), ((parallel_generatedRoot_233.w_15_0 ctx), ((parallel_generatedRoot_233.w_18_0 ctx), ((parallel_generatedRoot_233.w_21_0 ctx), ((parallel_generatedRoot_233.w_24_0 ctx), ())))))) ((parallel_generatedRoot_233.w_25_0 ctx), ((parallel_generatedRoot_233.w_25_1 ctx), ((parallel_generatedRoot_233.w_25_2 ctx), ((parallel_generatedRoot_233.w_25_3 ctx), ((parallel_generatedRoot_233.w_25_4 ctx), ((parallel_generatedRoot_233.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_233.outs ctx

theorem parallel_generatedRoot_233.facts_0  {ctx : parallel_generatedRoot_233Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_233.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_233.w_0_0 ctx) (parallel_generatedRoot_233.w_1_0 ctx) (parallel_generatedRoot_233.w_2_0 ctx) (parallel_generatedRoot_233.w_3_0 ctx) (parallel_generatedRoot_233.w_4_0 ctx) (parallel_generatedRoot_233.w_5_0 ctx) (parallel_generatedRoot_233.w_6_0 ctx) (parallel_generatedRoot_233.w_6_1 ctx) (parallel_generatedRoot_233.w_6_2 ctx) (parallel_generatedRoot_233.w_6_3 ctx) (parallel_generatedRoot_233.w_6_4 ctx) (parallel_generatedRoot_233.w_6_5 ctx) (parallel_generatedRoot_233.w_7_0 ctx) (parallel_generatedRoot_233.w_8_0 ctx) (parallel_generatedRoot_233.w_9_0 ctx) (parallel_generatedRoot_233.w_10_0 ctx) (parallel_generatedRoot_233.w_11_0 ctx) (parallel_generatedRoot_233.w_12_0 ctx) (parallel_generatedRoot_233.w_13_0 ctx) (parallel_generatedRoot_233.w_14_0 ctx) (parallel_generatedRoot_233.w_15_0 ctx) (parallel_generatedRoot_233.w_16_0 ctx) (parallel_generatedRoot_233.w_17_0 ctx) (parallel_generatedRoot_233.w_18_0 ctx) (parallel_generatedRoot_233.w_19_0 ctx) (parallel_generatedRoot_233.w_20_0 ctx) (parallel_generatedRoot_233.w_21_0 ctx) (parallel_generatedRoot_233.w_22_0 ctx) (parallel_generatedRoot_233.w_23_0 ctx) (parallel_generatedRoot_233.w_24_0 ctx) (parallel_generatedRoot_233.w_25_0 ctx) (parallel_generatedRoot_233.w_25_1 ctx) (parallel_generatedRoot_233.w_25_2 ctx) (parallel_generatedRoot_233.w_25_3 ctx) (parallel_generatedRoot_233.w_25_4 ctx) (parallel_generatedRoot_233.w_25_5 ctx) outputs) :
    parallel_generatedRoot_233.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_233.Facts  (ctx : parallel_generatedRoot_233Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_233.Facts_0 ctx outputs

theorem parallel_generatedRoot_233.facts  (ctx : parallel_generatedRoot_233Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_233.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_233.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_233.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_237.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_237.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_237Ctx" => (parallel_generatedRoot_237.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_237.w_0_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_237.w_1_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_237.w_2_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_237.w_3_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_237.w_4_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_237.w_5_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_237.w_6_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_237.w_6_1  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_237.w_6_2  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_237.w_6_3  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_237.w_6_4  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_237.w_6_5  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_237.w_7_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_237.w_8_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_237.w_9_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_7_0 ctx) (parallel_generatedRoot_237.w_8_0 ctx)

def parallel_generatedRoot_237.w_10_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_237.w_11_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_237.w_12_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_10_0 ctx) (parallel_generatedRoot_237.w_11_0 ctx)

def parallel_generatedRoot_237.w_13_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_237.w_14_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_237.w_15_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_13_0 ctx) (parallel_generatedRoot_237.w_14_0 ctx)

def parallel_generatedRoot_237.w_16_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_237.w_17_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_237.w_18_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_16_0 ctx) (parallel_generatedRoot_237.w_17_0 ctx)

def parallel_generatedRoot_237.w_19_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_237.w_20_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_237.w_21_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_19_0 ctx) (parallel_generatedRoot_237.w_20_0 ctx)

def parallel_generatedRoot_237.w_22_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_237.w_23_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_237.w_24_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_22_0 ctx) (parallel_generatedRoot_237.w_23_0 ctx)

def parallel_generatedRoot_237.w_25_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_237.w_25_1  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_237.w_25_2  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_237.w_25_3  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_237.w_25_4  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_237.w_25_5  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_237.outs  (ctx : parallel_generatedRoot_237Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_237.w_25_0 ctx), ((parallel_generatedRoot_237.w_25_1 ctx), ((parallel_generatedRoot_237.w_25_2 ctx), ((parallel_generatedRoot_237.w_25_3 ctx), ((parallel_generatedRoot_237.w_25_4 ctx), ((parallel_generatedRoot_237.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_237.Facts_0  (ctx : parallel_generatedRoot_237Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_237_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_237.w_0_0 ctx), ((parallel_generatedRoot_237.w_1_0 ctx), ((parallel_generatedRoot_237.w_2_0 ctx), ((parallel_generatedRoot_237.w_3_0 ctx), ((parallel_generatedRoot_237.w_4_0 ctx), ((parallel_generatedRoot_237.w_5_0 ctx), ())))))) (((parallel_generatedRoot_237.w_6_0 ctx) i), (((parallel_generatedRoot_237.w_6_1 ctx) i), (((parallel_generatedRoot_237.w_6_2 ctx) i), (((parallel_generatedRoot_237.w_6_3 ctx) i), (((parallel_generatedRoot_237.w_6_4 ctx) i), (((parallel_generatedRoot_237.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_0 ctx) (0) (parallel_generatedRoot_237.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_0 ctx) (1) (parallel_generatedRoot_237.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_1 ctx) (0) (parallel_generatedRoot_237.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_1 ctx) (1) (parallel_generatedRoot_237.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_2 ctx) (0) (parallel_generatedRoot_237.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_2 ctx) (1) (parallel_generatedRoot_237.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_3 ctx) (0) (parallel_generatedRoot_237.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_3 ctx) (1) (parallel_generatedRoot_237.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_4 ctx) (0) (parallel_generatedRoot_237.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_4 ctx) (1) (parallel_generatedRoot_237.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_5 ctx) (0) (parallel_generatedRoot_237.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_6_5 ctx) (1) (parallel_generatedRoot_237.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_237_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_237.w_6_0 ctx), ((parallel_generatedRoot_237.w_6_1 ctx), ((parallel_generatedRoot_237.w_6_2 ctx), ((parallel_generatedRoot_237.w_6_3 ctx), ((parallel_generatedRoot_237.w_6_4 ctx), ((parallel_generatedRoot_237.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_237.w_9_0 ctx), ((parallel_generatedRoot_237.w_12_0 ctx), ((parallel_generatedRoot_237.w_15_0 ctx), ((parallel_generatedRoot_237.w_18_0 ctx), ((parallel_generatedRoot_237.w_21_0 ctx), ((parallel_generatedRoot_237.w_24_0 ctx), ())))))) ((parallel_generatedRoot_237.w_25_0 ctx), ((parallel_generatedRoot_237.w_25_1 ctx), ((parallel_generatedRoot_237.w_25_2 ctx), ((parallel_generatedRoot_237.w_25_3 ctx), ((parallel_generatedRoot_237.w_25_4 ctx), ((parallel_generatedRoot_237.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_237.outs ctx

theorem parallel_generatedRoot_237.facts_0  {ctx : parallel_generatedRoot_237Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_237.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_237.w_0_0 ctx) (parallel_generatedRoot_237.w_1_0 ctx) (parallel_generatedRoot_237.w_2_0 ctx) (parallel_generatedRoot_237.w_3_0 ctx) (parallel_generatedRoot_237.w_4_0 ctx) (parallel_generatedRoot_237.w_5_0 ctx) (parallel_generatedRoot_237.w_6_0 ctx) (parallel_generatedRoot_237.w_6_1 ctx) (parallel_generatedRoot_237.w_6_2 ctx) (parallel_generatedRoot_237.w_6_3 ctx) (parallel_generatedRoot_237.w_6_4 ctx) (parallel_generatedRoot_237.w_6_5 ctx) (parallel_generatedRoot_237.w_7_0 ctx) (parallel_generatedRoot_237.w_8_0 ctx) (parallel_generatedRoot_237.w_9_0 ctx) (parallel_generatedRoot_237.w_10_0 ctx) (parallel_generatedRoot_237.w_11_0 ctx) (parallel_generatedRoot_237.w_12_0 ctx) (parallel_generatedRoot_237.w_13_0 ctx) (parallel_generatedRoot_237.w_14_0 ctx) (parallel_generatedRoot_237.w_15_0 ctx) (parallel_generatedRoot_237.w_16_0 ctx) (parallel_generatedRoot_237.w_17_0 ctx) (parallel_generatedRoot_237.w_18_0 ctx) (parallel_generatedRoot_237.w_19_0 ctx) (parallel_generatedRoot_237.w_20_0 ctx) (parallel_generatedRoot_237.w_21_0 ctx) (parallel_generatedRoot_237.w_22_0 ctx) (parallel_generatedRoot_237.w_23_0 ctx) (parallel_generatedRoot_237.w_24_0 ctx) (parallel_generatedRoot_237.w_25_0 ctx) (parallel_generatedRoot_237.w_25_1 ctx) (parallel_generatedRoot_237.w_25_2 ctx) (parallel_generatedRoot_237.w_25_3 ctx) (parallel_generatedRoot_237.w_25_4 ctx) (parallel_generatedRoot_237.w_25_5 ctx) outputs) :
    parallel_generatedRoot_237.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_237.Facts  (ctx : parallel_generatedRoot_237Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_237.Facts_0 ctx outputs

theorem parallel_generatedRoot_237.facts  (ctx : parallel_generatedRoot_237Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_237.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_237.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_237.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_241.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_241.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_241Ctx" => (parallel_generatedRoot_241.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_241.w_0_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_241.w_1_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_241.w_2_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_241.w_3_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_241.w_4_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_241.w_5_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_241.w_6_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_241.w_6_1  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_241.w_6_2  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_241.w_6_3  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_241.w_6_4  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_241.w_6_5  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_241.w_7_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_241.w_8_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_241.w_9_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_7_0 ctx) (parallel_generatedRoot_241.w_8_0 ctx)

def parallel_generatedRoot_241.w_10_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_241.w_11_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_241.w_12_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_10_0 ctx) (parallel_generatedRoot_241.w_11_0 ctx)

def parallel_generatedRoot_241.w_13_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_241.w_14_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_241.w_15_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_13_0 ctx) (parallel_generatedRoot_241.w_14_0 ctx)

def parallel_generatedRoot_241.w_16_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_241.w_17_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_241.w_18_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_16_0 ctx) (parallel_generatedRoot_241.w_17_0 ctx)

def parallel_generatedRoot_241.w_19_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_241.w_20_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_241.w_21_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_19_0 ctx) (parallel_generatedRoot_241.w_20_0 ctx)

def parallel_generatedRoot_241.w_22_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_241.w_23_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_241.w_24_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_22_0 ctx) (parallel_generatedRoot_241.w_23_0 ctx)

def parallel_generatedRoot_241.w_25_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_241.w_25_1  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_241.w_25_2  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_241.w_25_3  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_241.w_25_4  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_241.w_25_5  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_241.outs  (ctx : parallel_generatedRoot_241Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_241.w_25_0 ctx), ((parallel_generatedRoot_241.w_25_1 ctx), ((parallel_generatedRoot_241.w_25_2 ctx), ((parallel_generatedRoot_241.w_25_3 ctx), ((parallel_generatedRoot_241.w_25_4 ctx), ((parallel_generatedRoot_241.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_241.Facts_0  (ctx : parallel_generatedRoot_241Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_241_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_241.w_0_0 ctx), ((parallel_generatedRoot_241.w_1_0 ctx), ((parallel_generatedRoot_241.w_2_0 ctx), ((parallel_generatedRoot_241.w_3_0 ctx), ((parallel_generatedRoot_241.w_4_0 ctx), ((parallel_generatedRoot_241.w_5_0 ctx), ())))))) (((parallel_generatedRoot_241.w_6_0 ctx) i), (((parallel_generatedRoot_241.w_6_1 ctx) i), (((parallel_generatedRoot_241.w_6_2 ctx) i), (((parallel_generatedRoot_241.w_6_3 ctx) i), (((parallel_generatedRoot_241.w_6_4 ctx) i), (((parallel_generatedRoot_241.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_0 ctx) (0) (parallel_generatedRoot_241.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_0 ctx) (1) (parallel_generatedRoot_241.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_1 ctx) (0) (parallel_generatedRoot_241.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_1 ctx) (1) (parallel_generatedRoot_241.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_2 ctx) (0) (parallel_generatedRoot_241.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_2 ctx) (1) (parallel_generatedRoot_241.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_3 ctx) (0) (parallel_generatedRoot_241.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_3 ctx) (1) (parallel_generatedRoot_241.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_4 ctx) (0) (parallel_generatedRoot_241.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_4 ctx) (1) (parallel_generatedRoot_241.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_5 ctx) (0) (parallel_generatedRoot_241.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_6_5 ctx) (1) (parallel_generatedRoot_241.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_241_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_241.w_6_0 ctx), ((parallel_generatedRoot_241.w_6_1 ctx), ((parallel_generatedRoot_241.w_6_2 ctx), ((parallel_generatedRoot_241.w_6_3 ctx), ((parallel_generatedRoot_241.w_6_4 ctx), ((parallel_generatedRoot_241.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_241.w_9_0 ctx), ((parallel_generatedRoot_241.w_12_0 ctx), ((parallel_generatedRoot_241.w_15_0 ctx), ((parallel_generatedRoot_241.w_18_0 ctx), ((parallel_generatedRoot_241.w_21_0 ctx), ((parallel_generatedRoot_241.w_24_0 ctx), ())))))) ((parallel_generatedRoot_241.w_25_0 ctx), ((parallel_generatedRoot_241.w_25_1 ctx), ((parallel_generatedRoot_241.w_25_2 ctx), ((parallel_generatedRoot_241.w_25_3 ctx), ((parallel_generatedRoot_241.w_25_4 ctx), ((parallel_generatedRoot_241.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_241.outs ctx

theorem parallel_generatedRoot_241.facts_0  {ctx : parallel_generatedRoot_241Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_241.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_241.w_0_0 ctx) (parallel_generatedRoot_241.w_1_0 ctx) (parallel_generatedRoot_241.w_2_0 ctx) (parallel_generatedRoot_241.w_3_0 ctx) (parallel_generatedRoot_241.w_4_0 ctx) (parallel_generatedRoot_241.w_5_0 ctx) (parallel_generatedRoot_241.w_6_0 ctx) (parallel_generatedRoot_241.w_6_1 ctx) (parallel_generatedRoot_241.w_6_2 ctx) (parallel_generatedRoot_241.w_6_3 ctx) (parallel_generatedRoot_241.w_6_4 ctx) (parallel_generatedRoot_241.w_6_5 ctx) (parallel_generatedRoot_241.w_7_0 ctx) (parallel_generatedRoot_241.w_8_0 ctx) (parallel_generatedRoot_241.w_9_0 ctx) (parallel_generatedRoot_241.w_10_0 ctx) (parallel_generatedRoot_241.w_11_0 ctx) (parallel_generatedRoot_241.w_12_0 ctx) (parallel_generatedRoot_241.w_13_0 ctx) (parallel_generatedRoot_241.w_14_0 ctx) (parallel_generatedRoot_241.w_15_0 ctx) (parallel_generatedRoot_241.w_16_0 ctx) (parallel_generatedRoot_241.w_17_0 ctx) (parallel_generatedRoot_241.w_18_0 ctx) (parallel_generatedRoot_241.w_19_0 ctx) (parallel_generatedRoot_241.w_20_0 ctx) (parallel_generatedRoot_241.w_21_0 ctx) (parallel_generatedRoot_241.w_22_0 ctx) (parallel_generatedRoot_241.w_23_0 ctx) (parallel_generatedRoot_241.w_24_0 ctx) (parallel_generatedRoot_241.w_25_0 ctx) (parallel_generatedRoot_241.w_25_1 ctx) (parallel_generatedRoot_241.w_25_2 ctx) (parallel_generatedRoot_241.w_25_3 ctx) (parallel_generatedRoot_241.w_25_4 ctx) (parallel_generatedRoot_241.w_25_5 ctx) outputs) :
    parallel_generatedRoot_241.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_241.Facts  (ctx : parallel_generatedRoot_241Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_241.Facts_0 ctx outputs

theorem parallel_generatedRoot_241.facts  (ctx : parallel_generatedRoot_241Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_241.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_241.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_241.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_245.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_245.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_245Ctx" => (parallel_generatedRoot_245.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_245.w_0_0  (ctx : parallel_generatedRoot_245Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_245.w_1_0  (ctx : parallel_generatedRoot_245Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_245.w_2_0  (ctx : parallel_generatedRoot_245Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_245.w_3_0  (ctx : parallel_generatedRoot_245Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_245.w_4_0  (ctx : parallel_generatedRoot_245Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_245.w_5_0  (ctx : parallel_generatedRoot_245Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_245.w_6_0  (ctx : parallel_generatedRoot_245Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_245.w_6_1  (ctx : parallel_generatedRoot_245Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_245.w_6_2  (ctx : parallel_generatedRoot_245Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_245.w_6_3  (ctx : parallel_generatedRoot_245Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_245.w_6_4  (ctx : parallel_generatedRoot_245Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_245.w_6_5  (ctx : parallel_generatedRoot_245Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_245.w_7_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_245.w_8_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_245.w_9_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_245.w_7_0 ctx) (parallel_generatedRoot_245.w_8_0 ctx)

def parallel_generatedRoot_245.w_10_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_245.w_11_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_245.w_12_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_245.w_10_0 ctx) (parallel_generatedRoot_245.w_11_0 ctx)

def parallel_generatedRoot_245.w_13_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_245.w_14_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_245.w_15_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_245.w_13_0 ctx) (parallel_generatedRoot_245.w_14_0 ctx)

def parallel_generatedRoot_245.w_16_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_245.w_17_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_245.w_18_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_245.w_16_0 ctx) (parallel_generatedRoot_245.w_17_0 ctx)

def parallel_generatedRoot_245.w_19_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_245.w_20_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_245.w_21_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_245.w_19_0 ctx) (parallel_generatedRoot_245.w_20_0 ctx)

def parallel_generatedRoot_245.w_22_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_245.w_23_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_245.w_24_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_245.w_22_0 ctx) (parallel_generatedRoot_245.w_23_0 ctx)

def parallel_generatedRoot_245.w_25_0  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_245.w_25_1  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_245.w_25_2  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_245.w_25_3  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_245.w_25_4  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_245.w_25_5  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_245.outs  (ctx : parallel_generatedRoot_245Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_245.w_25_0 ctx), ((parallel_generatedRoot_245.w_25_1 ctx), ((parallel_generatedRoot_245.w_25_2 ctx), ((parallel_generatedRoot_245.w_25_3 ctx), ((parallel_generatedRoot_245.w_25_4 ctx), ((parallel_generatedRoot_245.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_245.Facts_0  (ctx : parallel_generatedRoot_245Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_245_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_245.w_0_0 ctx), ((parallel_generatedRoot_245.w_1_0 ctx), ((parallel_generatedRoot_245.w_2_0 ctx), ((parallel_generatedRoot_245.w_3_0 ctx), ((parallel_generatedRoot_245.w_4_0 ctx), ((parallel_generatedRoot_245.w_5_0 ctx), ())))))) (((parallel_generatedRoot_245.w_6_0 ctx) i), (((parallel_generatedRoot_245.w_6_1 ctx) i), (((parallel_generatedRoot_245.w_6_2 ctx) i), (((parallel_generatedRoot_245.w_6_3 ctx) i), (((parallel_generatedRoot_245.w_6_4 ctx) i), (((parallel_generatedRoot_245.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_0 ctx) (0) (parallel_generatedRoot_245.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_0 ctx) (1) (parallel_generatedRoot_245.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_1 ctx) (0) (parallel_generatedRoot_245.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_1 ctx) (1) (parallel_generatedRoot_245.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_2 ctx) (0) (parallel_generatedRoot_245.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_2 ctx) (1) (parallel_generatedRoot_245.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_3 ctx) (0) (parallel_generatedRoot_245.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_3 ctx) (1) (parallel_generatedRoot_245.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_4 ctx) (0) (parallel_generatedRoot_245.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_4 ctx) (1) (parallel_generatedRoot_245.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_5 ctx) (0) (parallel_generatedRoot_245.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_245.w_6_5 ctx) (1) (parallel_generatedRoot_245.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_245_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_245.w_6_0 ctx), ((parallel_generatedRoot_245.w_6_1 ctx), ((parallel_generatedRoot_245.w_6_2 ctx), ((parallel_generatedRoot_245.w_6_3 ctx), ((parallel_generatedRoot_245.w_6_4 ctx), ((parallel_generatedRoot_245.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_245.w_9_0 ctx), ((parallel_generatedRoot_245.w_12_0 ctx), ((parallel_generatedRoot_245.w_15_0 ctx), ((parallel_generatedRoot_245.w_18_0 ctx), ((parallel_generatedRoot_245.w_21_0 ctx), ((parallel_generatedRoot_245.w_24_0 ctx), ())))))) ((parallel_generatedRoot_245.w_25_0 ctx), ((parallel_generatedRoot_245.w_25_1 ctx), ((parallel_generatedRoot_245.w_25_2 ctx), ((parallel_generatedRoot_245.w_25_3 ctx), ((parallel_generatedRoot_245.w_25_4 ctx), ((parallel_generatedRoot_245.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_245.outs ctx

theorem parallel_generatedRoot_245.facts_0  {ctx : parallel_generatedRoot_245Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_245.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_245.w_0_0 ctx) (parallel_generatedRoot_245.w_1_0 ctx) (parallel_generatedRoot_245.w_2_0 ctx) (parallel_generatedRoot_245.w_3_0 ctx) (parallel_generatedRoot_245.w_4_0 ctx) (parallel_generatedRoot_245.w_5_0 ctx) (parallel_generatedRoot_245.w_6_0 ctx) (parallel_generatedRoot_245.w_6_1 ctx) (parallel_generatedRoot_245.w_6_2 ctx) (parallel_generatedRoot_245.w_6_3 ctx) (parallel_generatedRoot_245.w_6_4 ctx) (parallel_generatedRoot_245.w_6_5 ctx) (parallel_generatedRoot_245.w_7_0 ctx) (parallel_generatedRoot_245.w_8_0 ctx) (parallel_generatedRoot_245.w_9_0 ctx) (parallel_generatedRoot_245.w_10_0 ctx) (parallel_generatedRoot_245.w_11_0 ctx) (parallel_generatedRoot_245.w_12_0 ctx) (parallel_generatedRoot_245.w_13_0 ctx) (parallel_generatedRoot_245.w_14_0 ctx) (parallel_generatedRoot_245.w_15_0 ctx) (parallel_generatedRoot_245.w_16_0 ctx) (parallel_generatedRoot_245.w_17_0 ctx) (parallel_generatedRoot_245.w_18_0 ctx) (parallel_generatedRoot_245.w_19_0 ctx) (parallel_generatedRoot_245.w_20_0 ctx) (parallel_generatedRoot_245.w_21_0 ctx) (parallel_generatedRoot_245.w_22_0 ctx) (parallel_generatedRoot_245.w_23_0 ctx) (parallel_generatedRoot_245.w_24_0 ctx) (parallel_generatedRoot_245.w_25_0 ctx) (parallel_generatedRoot_245.w_25_1 ctx) (parallel_generatedRoot_245.w_25_2 ctx) (parallel_generatedRoot_245.w_25_3 ctx) (parallel_generatedRoot_245.w_25_4 ctx) (parallel_generatedRoot_245.w_25_5 ctx) outputs) :
    parallel_generatedRoot_245.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_245.Facts  (ctx : parallel_generatedRoot_245Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_245.Facts_0 ctx outputs

theorem parallel_generatedRoot_245.facts  (ctx : parallel_generatedRoot_245Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_245.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_245.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_245.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_249.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_249.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_249Ctx" => (parallel_generatedRoot_249.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_249.w_0_0  (ctx : parallel_generatedRoot_249Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_249.w_1_0  (ctx : parallel_generatedRoot_249Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_249.w_2_0  (ctx : parallel_generatedRoot_249Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_249.w_3_0  (ctx : parallel_generatedRoot_249Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_249.w_4_0  (ctx : parallel_generatedRoot_249Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_249.w_5_0  (ctx : parallel_generatedRoot_249Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_249.w_6_0  (ctx : parallel_generatedRoot_249Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_249.w_6_1  (ctx : parallel_generatedRoot_249Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_249.w_6_2  (ctx : parallel_generatedRoot_249Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_249.w_6_3  (ctx : parallel_generatedRoot_249Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_249.w_6_4  (ctx : parallel_generatedRoot_249Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_249.w_6_5  (ctx : parallel_generatedRoot_249Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_249.w_7_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_249.w_8_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_249.w_9_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_249.w_7_0 ctx) (parallel_generatedRoot_249.w_8_0 ctx)

def parallel_generatedRoot_249.w_10_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_249.w_11_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_249.w_12_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_249.w_10_0 ctx) (parallel_generatedRoot_249.w_11_0 ctx)

def parallel_generatedRoot_249.w_13_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_249.w_14_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_249.w_15_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_249.w_13_0 ctx) (parallel_generatedRoot_249.w_14_0 ctx)

def parallel_generatedRoot_249.w_16_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_249.w_17_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_249.w_18_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_249.w_16_0 ctx) (parallel_generatedRoot_249.w_17_0 ctx)

def parallel_generatedRoot_249.w_19_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_249.w_20_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_249.w_21_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_249.w_19_0 ctx) (parallel_generatedRoot_249.w_20_0 ctx)

def parallel_generatedRoot_249.w_22_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_249.w_23_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_249.w_24_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_249.w_22_0 ctx) (parallel_generatedRoot_249.w_23_0 ctx)

def parallel_generatedRoot_249.w_25_0  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_249.w_25_1  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_249.w_25_2  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_249.w_25_3  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_249.w_25_4  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_249.w_25_5  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_249.outs  (ctx : parallel_generatedRoot_249Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_249.w_25_0 ctx), ((parallel_generatedRoot_249.w_25_1 ctx), ((parallel_generatedRoot_249.w_25_2 ctx), ((parallel_generatedRoot_249.w_25_3 ctx), ((parallel_generatedRoot_249.w_25_4 ctx), ((parallel_generatedRoot_249.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_249.Facts_0  (ctx : parallel_generatedRoot_249Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_249_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_249.w_0_0 ctx), ((parallel_generatedRoot_249.w_1_0 ctx), ((parallel_generatedRoot_249.w_2_0 ctx), ((parallel_generatedRoot_249.w_3_0 ctx), ((parallel_generatedRoot_249.w_4_0 ctx), ((parallel_generatedRoot_249.w_5_0 ctx), ())))))) (((parallel_generatedRoot_249.w_6_0 ctx) i), (((parallel_generatedRoot_249.w_6_1 ctx) i), (((parallel_generatedRoot_249.w_6_2 ctx) i), (((parallel_generatedRoot_249.w_6_3 ctx) i), (((parallel_generatedRoot_249.w_6_4 ctx) i), (((parallel_generatedRoot_249.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_0 ctx) (0) (parallel_generatedRoot_249.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_0 ctx) (1) (parallel_generatedRoot_249.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_1 ctx) (0) (parallel_generatedRoot_249.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_1 ctx) (1) (parallel_generatedRoot_249.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_2 ctx) (0) (parallel_generatedRoot_249.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_2 ctx) (1) (parallel_generatedRoot_249.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_3 ctx) (0) (parallel_generatedRoot_249.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_3 ctx) (1) (parallel_generatedRoot_249.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_4 ctx) (0) (parallel_generatedRoot_249.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_4 ctx) (1) (parallel_generatedRoot_249.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_5 ctx) (0) (parallel_generatedRoot_249.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_249.w_6_5 ctx) (1) (parallel_generatedRoot_249.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_249_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_249.w_6_0 ctx), ((parallel_generatedRoot_249.w_6_1 ctx), ((parallel_generatedRoot_249.w_6_2 ctx), ((parallel_generatedRoot_249.w_6_3 ctx), ((parallel_generatedRoot_249.w_6_4 ctx), ((parallel_generatedRoot_249.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_249.w_9_0 ctx), ((parallel_generatedRoot_249.w_12_0 ctx), ((parallel_generatedRoot_249.w_15_0 ctx), ((parallel_generatedRoot_249.w_18_0 ctx), ((parallel_generatedRoot_249.w_21_0 ctx), ((parallel_generatedRoot_249.w_24_0 ctx), ())))))) ((parallel_generatedRoot_249.w_25_0 ctx), ((parallel_generatedRoot_249.w_25_1 ctx), ((parallel_generatedRoot_249.w_25_2 ctx), ((parallel_generatedRoot_249.w_25_3 ctx), ((parallel_generatedRoot_249.w_25_4 ctx), ((parallel_generatedRoot_249.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_249.outs ctx

theorem parallel_generatedRoot_249.facts_0  {ctx : parallel_generatedRoot_249Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_249.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_249.w_0_0 ctx) (parallel_generatedRoot_249.w_1_0 ctx) (parallel_generatedRoot_249.w_2_0 ctx) (parallel_generatedRoot_249.w_3_0 ctx) (parallel_generatedRoot_249.w_4_0 ctx) (parallel_generatedRoot_249.w_5_0 ctx) (parallel_generatedRoot_249.w_6_0 ctx) (parallel_generatedRoot_249.w_6_1 ctx) (parallel_generatedRoot_249.w_6_2 ctx) (parallel_generatedRoot_249.w_6_3 ctx) (parallel_generatedRoot_249.w_6_4 ctx) (parallel_generatedRoot_249.w_6_5 ctx) (parallel_generatedRoot_249.w_7_0 ctx) (parallel_generatedRoot_249.w_8_0 ctx) (parallel_generatedRoot_249.w_9_0 ctx) (parallel_generatedRoot_249.w_10_0 ctx) (parallel_generatedRoot_249.w_11_0 ctx) (parallel_generatedRoot_249.w_12_0 ctx) (parallel_generatedRoot_249.w_13_0 ctx) (parallel_generatedRoot_249.w_14_0 ctx) (parallel_generatedRoot_249.w_15_0 ctx) (parallel_generatedRoot_249.w_16_0 ctx) (parallel_generatedRoot_249.w_17_0 ctx) (parallel_generatedRoot_249.w_18_0 ctx) (parallel_generatedRoot_249.w_19_0 ctx) (parallel_generatedRoot_249.w_20_0 ctx) (parallel_generatedRoot_249.w_21_0 ctx) (parallel_generatedRoot_249.w_22_0 ctx) (parallel_generatedRoot_249.w_23_0 ctx) (parallel_generatedRoot_249.w_24_0 ctx) (parallel_generatedRoot_249.w_25_0 ctx) (parallel_generatedRoot_249.w_25_1 ctx) (parallel_generatedRoot_249.w_25_2 ctx) (parallel_generatedRoot_249.w_25_3 ctx) (parallel_generatedRoot_249.w_25_4 ctx) (parallel_generatedRoot_249.w_25_5 ctx) outputs) :
    parallel_generatedRoot_249.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_249.Facts  (ctx : parallel_generatedRoot_249Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_249.Facts_0 ctx outputs

theorem parallel_generatedRoot_249.facts  (ctx : parallel_generatedRoot_249Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_249.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_249.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_249.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_253.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_253.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_253Ctx" => (parallel_generatedRoot_253.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_253.w_0_0  (ctx : parallel_generatedRoot_253Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_253.w_1_0  (ctx : parallel_generatedRoot_253Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_253.w_2_0  (ctx : parallel_generatedRoot_253Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_253.w_3_0  (ctx : parallel_generatedRoot_253Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_253.w_4_0  (ctx : parallel_generatedRoot_253Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_253.w_5_0  (ctx : parallel_generatedRoot_253Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_253.w_6_0  (ctx : parallel_generatedRoot_253Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_253.w_6_1  (ctx : parallel_generatedRoot_253Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_253.w_6_2  (ctx : parallel_generatedRoot_253Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_253.w_6_3  (ctx : parallel_generatedRoot_253Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_253.w_6_4  (ctx : parallel_generatedRoot_253Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_253.w_6_5  (ctx : parallel_generatedRoot_253Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_253.w_7_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_253.w_8_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_253.w_9_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_253.w_7_0 ctx) (parallel_generatedRoot_253.w_8_0 ctx)

def parallel_generatedRoot_253.w_10_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_253.w_11_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_253.w_12_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_253.w_10_0 ctx) (parallel_generatedRoot_253.w_11_0 ctx)

def parallel_generatedRoot_253.w_13_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_253.w_14_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_253.w_15_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_253.w_13_0 ctx) (parallel_generatedRoot_253.w_14_0 ctx)

def parallel_generatedRoot_253.w_16_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_253.w_17_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_253.w_18_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_253.w_16_0 ctx) (parallel_generatedRoot_253.w_17_0 ctx)

def parallel_generatedRoot_253.w_19_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_253.w_20_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_253.w_21_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_253.w_19_0 ctx) (parallel_generatedRoot_253.w_20_0 ctx)

def parallel_generatedRoot_253.w_22_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_253.w_23_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_253.w_24_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_253.w_22_0 ctx) (parallel_generatedRoot_253.w_23_0 ctx)

def parallel_generatedRoot_253.w_25_0  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_253.w_25_1  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_253.w_25_2  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_253.w_25_3  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_253.w_25_4  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_253.w_25_5  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_253.outs  (ctx : parallel_generatedRoot_253Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_253.w_25_0 ctx), ((parallel_generatedRoot_253.w_25_1 ctx), ((parallel_generatedRoot_253.w_25_2 ctx), ((parallel_generatedRoot_253.w_25_3 ctx), ((parallel_generatedRoot_253.w_25_4 ctx), ((parallel_generatedRoot_253.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_253.Facts_0  (ctx : parallel_generatedRoot_253Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_253_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_253.w_0_0 ctx), ((parallel_generatedRoot_253.w_1_0 ctx), ((parallel_generatedRoot_253.w_2_0 ctx), ((parallel_generatedRoot_253.w_3_0 ctx), ((parallel_generatedRoot_253.w_4_0 ctx), ((parallel_generatedRoot_253.w_5_0 ctx), ())))))) (((parallel_generatedRoot_253.w_6_0 ctx) i), (((parallel_generatedRoot_253.w_6_1 ctx) i), (((parallel_generatedRoot_253.w_6_2 ctx) i), (((parallel_generatedRoot_253.w_6_3 ctx) i), (((parallel_generatedRoot_253.w_6_4 ctx) i), (((parallel_generatedRoot_253.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_0 ctx) (0) (parallel_generatedRoot_253.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_0 ctx) (1) (parallel_generatedRoot_253.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_1 ctx) (0) (parallel_generatedRoot_253.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_1 ctx) (1) (parallel_generatedRoot_253.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_2 ctx) (0) (parallel_generatedRoot_253.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_2 ctx) (1) (parallel_generatedRoot_253.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_3 ctx) (0) (parallel_generatedRoot_253.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_3 ctx) (1) (parallel_generatedRoot_253.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_4 ctx) (0) (parallel_generatedRoot_253.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_4 ctx) (1) (parallel_generatedRoot_253.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_5 ctx) (0) (parallel_generatedRoot_253.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_253.w_6_5 ctx) (1) (parallel_generatedRoot_253.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_253_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_253.w_6_0 ctx), ((parallel_generatedRoot_253.w_6_1 ctx), ((parallel_generatedRoot_253.w_6_2 ctx), ((parallel_generatedRoot_253.w_6_3 ctx), ((parallel_generatedRoot_253.w_6_4 ctx), ((parallel_generatedRoot_253.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_253.w_9_0 ctx), ((parallel_generatedRoot_253.w_12_0 ctx), ((parallel_generatedRoot_253.w_15_0 ctx), ((parallel_generatedRoot_253.w_18_0 ctx), ((parallel_generatedRoot_253.w_21_0 ctx), ((parallel_generatedRoot_253.w_24_0 ctx), ())))))) ((parallel_generatedRoot_253.w_25_0 ctx), ((parallel_generatedRoot_253.w_25_1 ctx), ((parallel_generatedRoot_253.w_25_2 ctx), ((parallel_generatedRoot_253.w_25_3 ctx), ((parallel_generatedRoot_253.w_25_4 ctx), ((parallel_generatedRoot_253.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_253.outs ctx

theorem parallel_generatedRoot_253.facts_0  {ctx : parallel_generatedRoot_253Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_253.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_253.w_0_0 ctx) (parallel_generatedRoot_253.w_1_0 ctx) (parallel_generatedRoot_253.w_2_0 ctx) (parallel_generatedRoot_253.w_3_0 ctx) (parallel_generatedRoot_253.w_4_0 ctx) (parallel_generatedRoot_253.w_5_0 ctx) (parallel_generatedRoot_253.w_6_0 ctx) (parallel_generatedRoot_253.w_6_1 ctx) (parallel_generatedRoot_253.w_6_2 ctx) (parallel_generatedRoot_253.w_6_3 ctx) (parallel_generatedRoot_253.w_6_4 ctx) (parallel_generatedRoot_253.w_6_5 ctx) (parallel_generatedRoot_253.w_7_0 ctx) (parallel_generatedRoot_253.w_8_0 ctx) (parallel_generatedRoot_253.w_9_0 ctx) (parallel_generatedRoot_253.w_10_0 ctx) (parallel_generatedRoot_253.w_11_0 ctx) (parallel_generatedRoot_253.w_12_0 ctx) (parallel_generatedRoot_253.w_13_0 ctx) (parallel_generatedRoot_253.w_14_0 ctx) (parallel_generatedRoot_253.w_15_0 ctx) (parallel_generatedRoot_253.w_16_0 ctx) (parallel_generatedRoot_253.w_17_0 ctx) (parallel_generatedRoot_253.w_18_0 ctx) (parallel_generatedRoot_253.w_19_0 ctx) (parallel_generatedRoot_253.w_20_0 ctx) (parallel_generatedRoot_253.w_21_0 ctx) (parallel_generatedRoot_253.w_22_0 ctx) (parallel_generatedRoot_253.w_23_0 ctx) (parallel_generatedRoot_253.w_24_0 ctx) (parallel_generatedRoot_253.w_25_0 ctx) (parallel_generatedRoot_253.w_25_1 ctx) (parallel_generatedRoot_253.w_25_2 ctx) (parallel_generatedRoot_253.w_25_3 ctx) (parallel_generatedRoot_253.w_25_4 ctx) (parallel_generatedRoot_253.w_25_5 ctx) outputs) :
    parallel_generatedRoot_253.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_253.Facts  (ctx : parallel_generatedRoot_253Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_253.Facts_0 ctx outputs

theorem parallel_generatedRoot_253.facts  (ctx : parallel_generatedRoot_253Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_253.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_253.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_253.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_257.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_257.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_257Ctx" => (parallel_generatedRoot_257.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_257.w_0_0  (ctx : parallel_generatedRoot_257Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_257.w_1_0  (ctx : parallel_generatedRoot_257Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_257.w_2_0  (ctx : parallel_generatedRoot_257Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_257.w_3_0  (ctx : parallel_generatedRoot_257Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_257.w_4_0  (ctx : parallel_generatedRoot_257Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_257.w_5_0  (ctx : parallel_generatedRoot_257Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_257.w_6_0  (ctx : parallel_generatedRoot_257Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_257.w_6_1  (ctx : parallel_generatedRoot_257Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_257.w_6_2  (ctx : parallel_generatedRoot_257Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_257.w_6_3  (ctx : parallel_generatedRoot_257Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_257.w_6_4  (ctx : parallel_generatedRoot_257Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_257.w_6_5  (ctx : parallel_generatedRoot_257Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_257.w_7_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_257.w_8_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_257.w_9_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_257.w_7_0 ctx) (parallel_generatedRoot_257.w_8_0 ctx)

def parallel_generatedRoot_257.w_10_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_257.w_11_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_257.w_12_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_257.w_10_0 ctx) (parallel_generatedRoot_257.w_11_0 ctx)

def parallel_generatedRoot_257.w_13_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_257.w_14_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_257.w_15_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_257.w_13_0 ctx) (parallel_generatedRoot_257.w_14_0 ctx)

def parallel_generatedRoot_257.w_16_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_257.w_17_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_257.w_18_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_257.w_16_0 ctx) (parallel_generatedRoot_257.w_17_0 ctx)

def parallel_generatedRoot_257.w_19_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_257.w_20_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_257.w_21_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_257.w_19_0 ctx) (parallel_generatedRoot_257.w_20_0 ctx)

def parallel_generatedRoot_257.w_22_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_257.w_23_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_257.w_24_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_257.w_22_0 ctx) (parallel_generatedRoot_257.w_23_0 ctx)

def parallel_generatedRoot_257.w_25_0  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_257.w_25_1  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_257.w_25_2  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_257.w_25_3  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_257.w_25_4  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_257.w_25_5  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_257.outs  (ctx : parallel_generatedRoot_257Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_257.w_25_0 ctx), ((parallel_generatedRoot_257.w_25_1 ctx), ((parallel_generatedRoot_257.w_25_2 ctx), ((parallel_generatedRoot_257.w_25_3 ctx), ((parallel_generatedRoot_257.w_25_4 ctx), ((parallel_generatedRoot_257.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_257.Facts_0  (ctx : parallel_generatedRoot_257Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_257_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_257.w_0_0 ctx), ((parallel_generatedRoot_257.w_1_0 ctx), ((parallel_generatedRoot_257.w_2_0 ctx), ((parallel_generatedRoot_257.w_3_0 ctx), ((parallel_generatedRoot_257.w_4_0 ctx), ((parallel_generatedRoot_257.w_5_0 ctx), ())))))) (((parallel_generatedRoot_257.w_6_0 ctx) i), (((parallel_generatedRoot_257.w_6_1 ctx) i), (((parallel_generatedRoot_257.w_6_2 ctx) i), (((parallel_generatedRoot_257.w_6_3 ctx) i), (((parallel_generatedRoot_257.w_6_4 ctx) i), (((parallel_generatedRoot_257.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_0 ctx) (0) (parallel_generatedRoot_257.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_0 ctx) (1) (parallel_generatedRoot_257.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_1 ctx) (0) (parallel_generatedRoot_257.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_1 ctx) (1) (parallel_generatedRoot_257.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_2 ctx) (0) (parallel_generatedRoot_257.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_2 ctx) (1) (parallel_generatedRoot_257.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_3 ctx) (0) (parallel_generatedRoot_257.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_3 ctx) (1) (parallel_generatedRoot_257.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_4 ctx) (0) (parallel_generatedRoot_257.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_4 ctx) (1) (parallel_generatedRoot_257.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_5 ctx) (0) (parallel_generatedRoot_257.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_257.w_6_5 ctx) (1) (parallel_generatedRoot_257.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_257_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_257.w_6_0 ctx), ((parallel_generatedRoot_257.w_6_1 ctx), ((parallel_generatedRoot_257.w_6_2 ctx), ((parallel_generatedRoot_257.w_6_3 ctx), ((parallel_generatedRoot_257.w_6_4 ctx), ((parallel_generatedRoot_257.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_257.w_9_0 ctx), ((parallel_generatedRoot_257.w_12_0 ctx), ((parallel_generatedRoot_257.w_15_0 ctx), ((parallel_generatedRoot_257.w_18_0 ctx), ((parallel_generatedRoot_257.w_21_0 ctx), ((parallel_generatedRoot_257.w_24_0 ctx), ())))))) ((parallel_generatedRoot_257.w_25_0 ctx), ((parallel_generatedRoot_257.w_25_1 ctx), ((parallel_generatedRoot_257.w_25_2 ctx), ((parallel_generatedRoot_257.w_25_3 ctx), ((parallel_generatedRoot_257.w_25_4 ctx), ((parallel_generatedRoot_257.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_257.outs ctx

theorem parallel_generatedRoot_257.facts_0  {ctx : parallel_generatedRoot_257Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_257.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_257.w_0_0 ctx) (parallel_generatedRoot_257.w_1_0 ctx) (parallel_generatedRoot_257.w_2_0 ctx) (parallel_generatedRoot_257.w_3_0 ctx) (parallel_generatedRoot_257.w_4_0 ctx) (parallel_generatedRoot_257.w_5_0 ctx) (parallel_generatedRoot_257.w_6_0 ctx) (parallel_generatedRoot_257.w_6_1 ctx) (parallel_generatedRoot_257.w_6_2 ctx) (parallel_generatedRoot_257.w_6_3 ctx) (parallel_generatedRoot_257.w_6_4 ctx) (parallel_generatedRoot_257.w_6_5 ctx) (parallel_generatedRoot_257.w_7_0 ctx) (parallel_generatedRoot_257.w_8_0 ctx) (parallel_generatedRoot_257.w_9_0 ctx) (parallel_generatedRoot_257.w_10_0 ctx) (parallel_generatedRoot_257.w_11_0 ctx) (parallel_generatedRoot_257.w_12_0 ctx) (parallel_generatedRoot_257.w_13_0 ctx) (parallel_generatedRoot_257.w_14_0 ctx) (parallel_generatedRoot_257.w_15_0 ctx) (parallel_generatedRoot_257.w_16_0 ctx) (parallel_generatedRoot_257.w_17_0 ctx) (parallel_generatedRoot_257.w_18_0 ctx) (parallel_generatedRoot_257.w_19_0 ctx) (parallel_generatedRoot_257.w_20_0 ctx) (parallel_generatedRoot_257.w_21_0 ctx) (parallel_generatedRoot_257.w_22_0 ctx) (parallel_generatedRoot_257.w_23_0 ctx) (parallel_generatedRoot_257.w_24_0 ctx) (parallel_generatedRoot_257.w_25_0 ctx) (parallel_generatedRoot_257.w_25_1 ctx) (parallel_generatedRoot_257.w_25_2 ctx) (parallel_generatedRoot_257.w_25_3 ctx) (parallel_generatedRoot_257.w_25_4 ctx) (parallel_generatedRoot_257.w_25_5 ctx) outputs) :
    parallel_generatedRoot_257.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_257.Facts  (ctx : parallel_generatedRoot_257Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_257.Facts_0 ctx outputs

theorem parallel_generatedRoot_257.facts  (ctx : parallel_generatedRoot_257Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_257.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_257.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_257.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_261.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_261.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_261Ctx" => (parallel_generatedRoot_261.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_261.w_0_0  (ctx : parallel_generatedRoot_261Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_261.w_1_0  (ctx : parallel_generatedRoot_261Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_261.w_2_0  (ctx : parallel_generatedRoot_261Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_261.w_3_0  (ctx : parallel_generatedRoot_261Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_261.w_4_0  (ctx : parallel_generatedRoot_261Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_261.w_5_0  (ctx : parallel_generatedRoot_261Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_261.w_6_0  (ctx : parallel_generatedRoot_261Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_261.w_6_1  (ctx : parallel_generatedRoot_261Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_261.w_6_2  (ctx : parallel_generatedRoot_261Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_261.w_6_3  (ctx : parallel_generatedRoot_261Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_261.w_6_4  (ctx : parallel_generatedRoot_261Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_261.w_6_5  (ctx : parallel_generatedRoot_261Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_261.w_7_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_261.w_8_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_261.w_9_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_261.w_7_0 ctx) (parallel_generatedRoot_261.w_8_0 ctx)

def parallel_generatedRoot_261.w_10_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_261.w_11_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_261.w_12_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_261.w_10_0 ctx) (parallel_generatedRoot_261.w_11_0 ctx)

def parallel_generatedRoot_261.w_13_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_261.w_14_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_261.w_15_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_261.w_13_0 ctx) (parallel_generatedRoot_261.w_14_0 ctx)

def parallel_generatedRoot_261.w_16_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_261.w_17_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_261.w_18_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_261.w_16_0 ctx) (parallel_generatedRoot_261.w_17_0 ctx)

def parallel_generatedRoot_261.w_19_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_261.w_20_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_261.w_21_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_261.w_19_0 ctx) (parallel_generatedRoot_261.w_20_0 ctx)

def parallel_generatedRoot_261.w_22_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_261.w_23_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_261.w_24_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_261.w_22_0 ctx) (parallel_generatedRoot_261.w_23_0 ctx)

def parallel_generatedRoot_261.w_25_0  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_261.w_25_1  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_261.w_25_2  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_261.w_25_3  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_261.w_25_4  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_261.w_25_5  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_261.outs  (ctx : parallel_generatedRoot_261Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_261.w_25_0 ctx), ((parallel_generatedRoot_261.w_25_1 ctx), ((parallel_generatedRoot_261.w_25_2 ctx), ((parallel_generatedRoot_261.w_25_3 ctx), ((parallel_generatedRoot_261.w_25_4 ctx), ((parallel_generatedRoot_261.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_261.Facts_0  (ctx : parallel_generatedRoot_261Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_261_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_261.w_0_0 ctx), ((parallel_generatedRoot_261.w_1_0 ctx), ((parallel_generatedRoot_261.w_2_0 ctx), ((parallel_generatedRoot_261.w_3_0 ctx), ((parallel_generatedRoot_261.w_4_0 ctx), ((parallel_generatedRoot_261.w_5_0 ctx), ())))))) (((parallel_generatedRoot_261.w_6_0 ctx) i), (((parallel_generatedRoot_261.w_6_1 ctx) i), (((parallel_generatedRoot_261.w_6_2 ctx) i), (((parallel_generatedRoot_261.w_6_3 ctx) i), (((parallel_generatedRoot_261.w_6_4 ctx) i), (((parallel_generatedRoot_261.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_0 ctx) (0) (parallel_generatedRoot_261.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_0 ctx) (1) (parallel_generatedRoot_261.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_1 ctx) (0) (parallel_generatedRoot_261.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_1 ctx) (1) (parallel_generatedRoot_261.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_2 ctx) (0) (parallel_generatedRoot_261.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_2 ctx) (1) (parallel_generatedRoot_261.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_3 ctx) (0) (parallel_generatedRoot_261.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_3 ctx) (1) (parallel_generatedRoot_261.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_4 ctx) (0) (parallel_generatedRoot_261.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_4 ctx) (1) (parallel_generatedRoot_261.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_5 ctx) (0) (parallel_generatedRoot_261.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_261.w_6_5 ctx) (1) (parallel_generatedRoot_261.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_261_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_261.w_6_0 ctx), ((parallel_generatedRoot_261.w_6_1 ctx), ((parallel_generatedRoot_261.w_6_2 ctx), ((parallel_generatedRoot_261.w_6_3 ctx), ((parallel_generatedRoot_261.w_6_4 ctx), ((parallel_generatedRoot_261.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_261.w_9_0 ctx), ((parallel_generatedRoot_261.w_12_0 ctx), ((parallel_generatedRoot_261.w_15_0 ctx), ((parallel_generatedRoot_261.w_18_0 ctx), ((parallel_generatedRoot_261.w_21_0 ctx), ((parallel_generatedRoot_261.w_24_0 ctx), ())))))) ((parallel_generatedRoot_261.w_25_0 ctx), ((parallel_generatedRoot_261.w_25_1 ctx), ((parallel_generatedRoot_261.w_25_2 ctx), ((parallel_generatedRoot_261.w_25_3 ctx), ((parallel_generatedRoot_261.w_25_4 ctx), ((parallel_generatedRoot_261.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_261.outs ctx

theorem parallel_generatedRoot_261.facts_0  {ctx : parallel_generatedRoot_261Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_261.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_261.w_0_0 ctx) (parallel_generatedRoot_261.w_1_0 ctx) (parallel_generatedRoot_261.w_2_0 ctx) (parallel_generatedRoot_261.w_3_0 ctx) (parallel_generatedRoot_261.w_4_0 ctx) (parallel_generatedRoot_261.w_5_0 ctx) (parallel_generatedRoot_261.w_6_0 ctx) (parallel_generatedRoot_261.w_6_1 ctx) (parallel_generatedRoot_261.w_6_2 ctx) (parallel_generatedRoot_261.w_6_3 ctx) (parallel_generatedRoot_261.w_6_4 ctx) (parallel_generatedRoot_261.w_6_5 ctx) (parallel_generatedRoot_261.w_7_0 ctx) (parallel_generatedRoot_261.w_8_0 ctx) (parallel_generatedRoot_261.w_9_0 ctx) (parallel_generatedRoot_261.w_10_0 ctx) (parallel_generatedRoot_261.w_11_0 ctx) (parallel_generatedRoot_261.w_12_0 ctx) (parallel_generatedRoot_261.w_13_0 ctx) (parallel_generatedRoot_261.w_14_0 ctx) (parallel_generatedRoot_261.w_15_0 ctx) (parallel_generatedRoot_261.w_16_0 ctx) (parallel_generatedRoot_261.w_17_0 ctx) (parallel_generatedRoot_261.w_18_0 ctx) (parallel_generatedRoot_261.w_19_0 ctx) (parallel_generatedRoot_261.w_20_0 ctx) (parallel_generatedRoot_261.w_21_0 ctx) (parallel_generatedRoot_261.w_22_0 ctx) (parallel_generatedRoot_261.w_23_0 ctx) (parallel_generatedRoot_261.w_24_0 ctx) (parallel_generatedRoot_261.w_25_0 ctx) (parallel_generatedRoot_261.w_25_1 ctx) (parallel_generatedRoot_261.w_25_2 ctx) (parallel_generatedRoot_261.w_25_3 ctx) (parallel_generatedRoot_261.w_25_4 ctx) (parallel_generatedRoot_261.w_25_5 ctx) outputs) :
    parallel_generatedRoot_261.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_261.Facts  (ctx : parallel_generatedRoot_261Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_261.Facts_0 ctx outputs

theorem parallel_generatedRoot_261.facts  (ctx : parallel_generatedRoot_261Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_261.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_261.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_261.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_265.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_265.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_265Ctx" => (parallel_generatedRoot_265.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_265.w_0_0  (ctx : parallel_generatedRoot_265Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_265.w_1_0  (ctx : parallel_generatedRoot_265Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_265.w_2_0  (ctx : parallel_generatedRoot_265Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_265.w_3_0  (ctx : parallel_generatedRoot_265Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_265.w_4_0  (ctx : parallel_generatedRoot_265Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_265.w_5_0  (ctx : parallel_generatedRoot_265Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_265.w_6_0  (ctx : parallel_generatedRoot_265Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_265.w_6_1  (ctx : parallel_generatedRoot_265Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_265.w_6_2  (ctx : parallel_generatedRoot_265Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_265.w_6_3  (ctx : parallel_generatedRoot_265Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_265.w_6_4  (ctx : parallel_generatedRoot_265Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_265.w_6_5  (ctx : parallel_generatedRoot_265Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_265.w_7_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_265.w_8_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_265.w_9_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_265.w_7_0 ctx) (parallel_generatedRoot_265.w_8_0 ctx)

def parallel_generatedRoot_265.w_10_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_265.w_11_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_265.w_12_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_265.w_10_0 ctx) (parallel_generatedRoot_265.w_11_0 ctx)

def parallel_generatedRoot_265.w_13_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_265.w_14_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_265.w_15_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_265.w_13_0 ctx) (parallel_generatedRoot_265.w_14_0 ctx)

def parallel_generatedRoot_265.w_16_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_265.w_17_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_265.w_18_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_265.w_16_0 ctx) (parallel_generatedRoot_265.w_17_0 ctx)

def parallel_generatedRoot_265.w_19_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_265.w_20_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_265.w_21_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_265.w_19_0 ctx) (parallel_generatedRoot_265.w_20_0 ctx)

def parallel_generatedRoot_265.w_22_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_265.w_23_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_265.w_24_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_265.w_22_0 ctx) (parallel_generatedRoot_265.w_23_0 ctx)

def parallel_generatedRoot_265.w_25_0  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_265.w_25_1  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_265.w_25_2  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_265.w_25_3  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_265.w_25_4  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_265.w_25_5  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_265.outs  (ctx : parallel_generatedRoot_265Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_265.w_25_0 ctx), ((parallel_generatedRoot_265.w_25_1 ctx), ((parallel_generatedRoot_265.w_25_2 ctx), ((parallel_generatedRoot_265.w_25_3 ctx), ((parallel_generatedRoot_265.w_25_4 ctx), ((parallel_generatedRoot_265.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_265.Facts_0  (ctx : parallel_generatedRoot_265Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_265_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_265.w_0_0 ctx), ((parallel_generatedRoot_265.w_1_0 ctx), ((parallel_generatedRoot_265.w_2_0 ctx), ((parallel_generatedRoot_265.w_3_0 ctx), ((parallel_generatedRoot_265.w_4_0 ctx), ((parallel_generatedRoot_265.w_5_0 ctx), ())))))) (((parallel_generatedRoot_265.w_6_0 ctx) i), (((parallel_generatedRoot_265.w_6_1 ctx) i), (((parallel_generatedRoot_265.w_6_2 ctx) i), (((parallel_generatedRoot_265.w_6_3 ctx) i), (((parallel_generatedRoot_265.w_6_4 ctx) i), (((parallel_generatedRoot_265.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_0 ctx) (0) (parallel_generatedRoot_265.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_0 ctx) (1) (parallel_generatedRoot_265.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_1 ctx) (0) (parallel_generatedRoot_265.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_1 ctx) (1) (parallel_generatedRoot_265.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_2 ctx) (0) (parallel_generatedRoot_265.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_2 ctx) (1) (parallel_generatedRoot_265.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_3 ctx) (0) (parallel_generatedRoot_265.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_3 ctx) (1) (parallel_generatedRoot_265.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_4 ctx) (0) (parallel_generatedRoot_265.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_4 ctx) (1) (parallel_generatedRoot_265.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_5 ctx) (0) (parallel_generatedRoot_265.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_265.w_6_5 ctx) (1) (parallel_generatedRoot_265.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_265_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_265.w_6_0 ctx), ((parallel_generatedRoot_265.w_6_1 ctx), ((parallel_generatedRoot_265.w_6_2 ctx), ((parallel_generatedRoot_265.w_6_3 ctx), ((parallel_generatedRoot_265.w_6_4 ctx), ((parallel_generatedRoot_265.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_265.w_9_0 ctx), ((parallel_generatedRoot_265.w_12_0 ctx), ((parallel_generatedRoot_265.w_15_0 ctx), ((parallel_generatedRoot_265.w_18_0 ctx), ((parallel_generatedRoot_265.w_21_0 ctx), ((parallel_generatedRoot_265.w_24_0 ctx), ())))))) ((parallel_generatedRoot_265.w_25_0 ctx), ((parallel_generatedRoot_265.w_25_1 ctx), ((parallel_generatedRoot_265.w_25_2 ctx), ((parallel_generatedRoot_265.w_25_3 ctx), ((parallel_generatedRoot_265.w_25_4 ctx), ((parallel_generatedRoot_265.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_265.outs ctx

theorem parallel_generatedRoot_265.facts_0  {ctx : parallel_generatedRoot_265Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_265.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_265.w_0_0 ctx) (parallel_generatedRoot_265.w_1_0 ctx) (parallel_generatedRoot_265.w_2_0 ctx) (parallel_generatedRoot_265.w_3_0 ctx) (parallel_generatedRoot_265.w_4_0 ctx) (parallel_generatedRoot_265.w_5_0 ctx) (parallel_generatedRoot_265.w_6_0 ctx) (parallel_generatedRoot_265.w_6_1 ctx) (parallel_generatedRoot_265.w_6_2 ctx) (parallel_generatedRoot_265.w_6_3 ctx) (parallel_generatedRoot_265.w_6_4 ctx) (parallel_generatedRoot_265.w_6_5 ctx) (parallel_generatedRoot_265.w_7_0 ctx) (parallel_generatedRoot_265.w_8_0 ctx) (parallel_generatedRoot_265.w_9_0 ctx) (parallel_generatedRoot_265.w_10_0 ctx) (parallel_generatedRoot_265.w_11_0 ctx) (parallel_generatedRoot_265.w_12_0 ctx) (parallel_generatedRoot_265.w_13_0 ctx) (parallel_generatedRoot_265.w_14_0 ctx) (parallel_generatedRoot_265.w_15_0 ctx) (parallel_generatedRoot_265.w_16_0 ctx) (parallel_generatedRoot_265.w_17_0 ctx) (parallel_generatedRoot_265.w_18_0 ctx) (parallel_generatedRoot_265.w_19_0 ctx) (parallel_generatedRoot_265.w_20_0 ctx) (parallel_generatedRoot_265.w_21_0 ctx) (parallel_generatedRoot_265.w_22_0 ctx) (parallel_generatedRoot_265.w_23_0 ctx) (parallel_generatedRoot_265.w_24_0 ctx) (parallel_generatedRoot_265.w_25_0 ctx) (parallel_generatedRoot_265.w_25_1 ctx) (parallel_generatedRoot_265.w_25_2 ctx) (parallel_generatedRoot_265.w_25_3 ctx) (parallel_generatedRoot_265.w_25_4 ctx) (parallel_generatedRoot_265.w_25_5 ctx) outputs) :
    parallel_generatedRoot_265.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_265.Facts  (ctx : parallel_generatedRoot_265Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_265.Facts_0 ctx outputs

theorem parallel_generatedRoot_265.facts  (ctx : parallel_generatedRoot_265Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_265.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_265.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_265.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_269.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_269.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_269Ctx" => (parallel_generatedRoot_269.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_269.w_0_0  (ctx : parallel_generatedRoot_269Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_269.w_1_0  (ctx : parallel_generatedRoot_269Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_269.w_2_0  (ctx : parallel_generatedRoot_269Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_269.w_3_0  (ctx : parallel_generatedRoot_269Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_269.w_4_0  (ctx : parallel_generatedRoot_269Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_269.w_5_0  (ctx : parallel_generatedRoot_269Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_269.w_6_0  (ctx : parallel_generatedRoot_269Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_269.w_6_1  (ctx : parallel_generatedRoot_269Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_269.w_6_2  (ctx : parallel_generatedRoot_269Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_269.w_6_3  (ctx : parallel_generatedRoot_269Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_269.w_6_4  (ctx : parallel_generatedRoot_269Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_269.w_6_5  (ctx : parallel_generatedRoot_269Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_269.w_7_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_269.w_8_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_269.w_9_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_269.w_7_0 ctx) (parallel_generatedRoot_269.w_8_0 ctx)

def parallel_generatedRoot_269.w_10_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_269.w_11_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_269.w_12_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_269.w_10_0 ctx) (parallel_generatedRoot_269.w_11_0 ctx)

def parallel_generatedRoot_269.w_13_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_269.w_14_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_269.w_15_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_269.w_13_0 ctx) (parallel_generatedRoot_269.w_14_0 ctx)

def parallel_generatedRoot_269.w_16_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_269.w_17_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_269.w_18_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_269.w_16_0 ctx) (parallel_generatedRoot_269.w_17_0 ctx)

def parallel_generatedRoot_269.w_19_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_269.w_20_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_269.w_21_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_269.w_19_0 ctx) (parallel_generatedRoot_269.w_20_0 ctx)

def parallel_generatedRoot_269.w_22_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_269.w_23_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_269.w_24_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_269.w_22_0 ctx) (parallel_generatedRoot_269.w_23_0 ctx)

def parallel_generatedRoot_269.w_25_0  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_269.w_25_1  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_269.w_25_2  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_269.w_25_3  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_269.w_25_4  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_269.w_25_5  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_269.outs  (ctx : parallel_generatedRoot_269Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_269.w_25_0 ctx), ((parallel_generatedRoot_269.w_25_1 ctx), ((parallel_generatedRoot_269.w_25_2 ctx), ((parallel_generatedRoot_269.w_25_3 ctx), ((parallel_generatedRoot_269.w_25_4 ctx), ((parallel_generatedRoot_269.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_269.Facts_0  (ctx : parallel_generatedRoot_269Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_269_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_269.w_0_0 ctx), ((parallel_generatedRoot_269.w_1_0 ctx), ((parallel_generatedRoot_269.w_2_0 ctx), ((parallel_generatedRoot_269.w_3_0 ctx), ((parallel_generatedRoot_269.w_4_0 ctx), ((parallel_generatedRoot_269.w_5_0 ctx), ())))))) (((parallel_generatedRoot_269.w_6_0 ctx) i), (((parallel_generatedRoot_269.w_6_1 ctx) i), (((parallel_generatedRoot_269.w_6_2 ctx) i), (((parallel_generatedRoot_269.w_6_3 ctx) i), (((parallel_generatedRoot_269.w_6_4 ctx) i), (((parallel_generatedRoot_269.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_0 ctx) (0) (parallel_generatedRoot_269.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_0 ctx) (1) (parallel_generatedRoot_269.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_1 ctx) (0) (parallel_generatedRoot_269.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_1 ctx) (1) (parallel_generatedRoot_269.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_2 ctx) (0) (parallel_generatedRoot_269.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_2 ctx) (1) (parallel_generatedRoot_269.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_3 ctx) (0) (parallel_generatedRoot_269.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_3 ctx) (1) (parallel_generatedRoot_269.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_4 ctx) (0) (parallel_generatedRoot_269.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_4 ctx) (1) (parallel_generatedRoot_269.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_5 ctx) (0) (parallel_generatedRoot_269.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_269.w_6_5 ctx) (1) (parallel_generatedRoot_269.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_269_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_269.w_6_0 ctx), ((parallel_generatedRoot_269.w_6_1 ctx), ((parallel_generatedRoot_269.w_6_2 ctx), ((parallel_generatedRoot_269.w_6_3 ctx), ((parallel_generatedRoot_269.w_6_4 ctx), ((parallel_generatedRoot_269.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_269.w_9_0 ctx), ((parallel_generatedRoot_269.w_12_0 ctx), ((parallel_generatedRoot_269.w_15_0 ctx), ((parallel_generatedRoot_269.w_18_0 ctx), ((parallel_generatedRoot_269.w_21_0 ctx), ((parallel_generatedRoot_269.w_24_0 ctx), ())))))) ((parallel_generatedRoot_269.w_25_0 ctx), ((parallel_generatedRoot_269.w_25_1 ctx), ((parallel_generatedRoot_269.w_25_2 ctx), ((parallel_generatedRoot_269.w_25_3 ctx), ((parallel_generatedRoot_269.w_25_4 ctx), ((parallel_generatedRoot_269.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_269.outs ctx

theorem parallel_generatedRoot_269.facts_0  {ctx : parallel_generatedRoot_269Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_269.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_269.w_0_0 ctx) (parallel_generatedRoot_269.w_1_0 ctx) (parallel_generatedRoot_269.w_2_0 ctx) (parallel_generatedRoot_269.w_3_0 ctx) (parallel_generatedRoot_269.w_4_0 ctx) (parallel_generatedRoot_269.w_5_0 ctx) (parallel_generatedRoot_269.w_6_0 ctx) (parallel_generatedRoot_269.w_6_1 ctx) (parallel_generatedRoot_269.w_6_2 ctx) (parallel_generatedRoot_269.w_6_3 ctx) (parallel_generatedRoot_269.w_6_4 ctx) (parallel_generatedRoot_269.w_6_5 ctx) (parallel_generatedRoot_269.w_7_0 ctx) (parallel_generatedRoot_269.w_8_0 ctx) (parallel_generatedRoot_269.w_9_0 ctx) (parallel_generatedRoot_269.w_10_0 ctx) (parallel_generatedRoot_269.w_11_0 ctx) (parallel_generatedRoot_269.w_12_0 ctx) (parallel_generatedRoot_269.w_13_0 ctx) (parallel_generatedRoot_269.w_14_0 ctx) (parallel_generatedRoot_269.w_15_0 ctx) (parallel_generatedRoot_269.w_16_0 ctx) (parallel_generatedRoot_269.w_17_0 ctx) (parallel_generatedRoot_269.w_18_0 ctx) (parallel_generatedRoot_269.w_19_0 ctx) (parallel_generatedRoot_269.w_20_0 ctx) (parallel_generatedRoot_269.w_21_0 ctx) (parallel_generatedRoot_269.w_22_0 ctx) (parallel_generatedRoot_269.w_23_0 ctx) (parallel_generatedRoot_269.w_24_0 ctx) (parallel_generatedRoot_269.w_25_0 ctx) (parallel_generatedRoot_269.w_25_1 ctx) (parallel_generatedRoot_269.w_25_2 ctx) (parallel_generatedRoot_269.w_25_3 ctx) (parallel_generatedRoot_269.w_25_4 ctx) (parallel_generatedRoot_269.w_25_5 ctx) outputs) :
    parallel_generatedRoot_269.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_269.Facts  (ctx : parallel_generatedRoot_269Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_269.Facts_0 ctx outputs

theorem parallel_generatedRoot_269.facts  (ctx : parallel_generatedRoot_269Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_269.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_269.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_269.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_273.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_273.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_273Ctx" => (parallel_generatedRoot_273.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_273.w_0_0  (ctx : parallel_generatedRoot_273Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_273.w_1_0  (ctx : parallel_generatedRoot_273Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_273.w_2_0  (ctx : parallel_generatedRoot_273Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_273.w_3_0  (ctx : parallel_generatedRoot_273Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_273.w_4_0  (ctx : parallel_generatedRoot_273Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_273.w_5_0  (ctx : parallel_generatedRoot_273Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_273.w_6_0  (ctx : parallel_generatedRoot_273Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_273.w_6_1  (ctx : parallel_generatedRoot_273Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_273.w_6_2  (ctx : parallel_generatedRoot_273Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_273.w_6_3  (ctx : parallel_generatedRoot_273Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_273.w_6_4  (ctx : parallel_generatedRoot_273Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_273.w_6_5  (ctx : parallel_generatedRoot_273Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_273.w_7_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_273.w_8_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_273.w_9_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_273.w_7_0 ctx) (parallel_generatedRoot_273.w_8_0 ctx)

def parallel_generatedRoot_273.w_10_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_273.w_11_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_273.w_12_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_273.w_10_0 ctx) (parallel_generatedRoot_273.w_11_0 ctx)

def parallel_generatedRoot_273.w_13_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_273.w_14_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_273.w_15_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_273.w_13_0 ctx) (parallel_generatedRoot_273.w_14_0 ctx)

def parallel_generatedRoot_273.w_16_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_273.w_17_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_273.w_18_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_273.w_16_0 ctx) (parallel_generatedRoot_273.w_17_0 ctx)

def parallel_generatedRoot_273.w_19_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_273.w_20_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_273.w_21_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_273.w_19_0 ctx) (parallel_generatedRoot_273.w_20_0 ctx)

def parallel_generatedRoot_273.w_22_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_273.w_23_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_273.w_24_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_273.w_22_0 ctx) (parallel_generatedRoot_273.w_23_0 ctx)

def parallel_generatedRoot_273.w_25_0  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_273.w_25_1  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_273.w_25_2  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_273.w_25_3  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_273.w_25_4  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_273.w_25_5  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_273.outs  (ctx : parallel_generatedRoot_273Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_273.w_25_0 ctx), ((parallel_generatedRoot_273.w_25_1 ctx), ((parallel_generatedRoot_273.w_25_2 ctx), ((parallel_generatedRoot_273.w_25_3 ctx), ((parallel_generatedRoot_273.w_25_4 ctx), ((parallel_generatedRoot_273.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_273.Facts_0  (ctx : parallel_generatedRoot_273Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_273_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_273.w_0_0 ctx), ((parallel_generatedRoot_273.w_1_0 ctx), ((parallel_generatedRoot_273.w_2_0 ctx), ((parallel_generatedRoot_273.w_3_0 ctx), ((parallel_generatedRoot_273.w_4_0 ctx), ((parallel_generatedRoot_273.w_5_0 ctx), ())))))) (((parallel_generatedRoot_273.w_6_0 ctx) i), (((parallel_generatedRoot_273.w_6_1 ctx) i), (((parallel_generatedRoot_273.w_6_2 ctx) i), (((parallel_generatedRoot_273.w_6_3 ctx) i), (((parallel_generatedRoot_273.w_6_4 ctx) i), (((parallel_generatedRoot_273.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_0 ctx) (0) (parallel_generatedRoot_273.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_0 ctx) (1) (parallel_generatedRoot_273.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_1 ctx) (0) (parallel_generatedRoot_273.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_1 ctx) (1) (parallel_generatedRoot_273.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_2 ctx) (0) (parallel_generatedRoot_273.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_2 ctx) (1) (parallel_generatedRoot_273.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_3 ctx) (0) (parallel_generatedRoot_273.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_3 ctx) (1) (parallel_generatedRoot_273.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_4 ctx) (0) (parallel_generatedRoot_273.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_4 ctx) (1) (parallel_generatedRoot_273.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_5 ctx) (0) (parallel_generatedRoot_273.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_273.w_6_5 ctx) (1) (parallel_generatedRoot_273.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_273_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_273.w_6_0 ctx), ((parallel_generatedRoot_273.w_6_1 ctx), ((parallel_generatedRoot_273.w_6_2 ctx), ((parallel_generatedRoot_273.w_6_3 ctx), ((parallel_generatedRoot_273.w_6_4 ctx), ((parallel_generatedRoot_273.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_273.w_9_0 ctx), ((parallel_generatedRoot_273.w_12_0 ctx), ((parallel_generatedRoot_273.w_15_0 ctx), ((parallel_generatedRoot_273.w_18_0 ctx), ((parallel_generatedRoot_273.w_21_0 ctx), ((parallel_generatedRoot_273.w_24_0 ctx), ())))))) ((parallel_generatedRoot_273.w_25_0 ctx), ((parallel_generatedRoot_273.w_25_1 ctx), ((parallel_generatedRoot_273.w_25_2 ctx), ((parallel_generatedRoot_273.w_25_3 ctx), ((parallel_generatedRoot_273.w_25_4 ctx), ((parallel_generatedRoot_273.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_273.outs ctx

theorem parallel_generatedRoot_273.facts_0  {ctx : parallel_generatedRoot_273Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_273.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_273.w_0_0 ctx) (parallel_generatedRoot_273.w_1_0 ctx) (parallel_generatedRoot_273.w_2_0 ctx) (parallel_generatedRoot_273.w_3_0 ctx) (parallel_generatedRoot_273.w_4_0 ctx) (parallel_generatedRoot_273.w_5_0 ctx) (parallel_generatedRoot_273.w_6_0 ctx) (parallel_generatedRoot_273.w_6_1 ctx) (parallel_generatedRoot_273.w_6_2 ctx) (parallel_generatedRoot_273.w_6_3 ctx) (parallel_generatedRoot_273.w_6_4 ctx) (parallel_generatedRoot_273.w_6_5 ctx) (parallel_generatedRoot_273.w_7_0 ctx) (parallel_generatedRoot_273.w_8_0 ctx) (parallel_generatedRoot_273.w_9_0 ctx) (parallel_generatedRoot_273.w_10_0 ctx) (parallel_generatedRoot_273.w_11_0 ctx) (parallel_generatedRoot_273.w_12_0 ctx) (parallel_generatedRoot_273.w_13_0 ctx) (parallel_generatedRoot_273.w_14_0 ctx) (parallel_generatedRoot_273.w_15_0 ctx) (parallel_generatedRoot_273.w_16_0 ctx) (parallel_generatedRoot_273.w_17_0 ctx) (parallel_generatedRoot_273.w_18_0 ctx) (parallel_generatedRoot_273.w_19_0 ctx) (parallel_generatedRoot_273.w_20_0 ctx) (parallel_generatedRoot_273.w_21_0 ctx) (parallel_generatedRoot_273.w_22_0 ctx) (parallel_generatedRoot_273.w_23_0 ctx) (parallel_generatedRoot_273.w_24_0 ctx) (parallel_generatedRoot_273.w_25_0 ctx) (parallel_generatedRoot_273.w_25_1 ctx) (parallel_generatedRoot_273.w_25_2 ctx) (parallel_generatedRoot_273.w_25_3 ctx) (parallel_generatedRoot_273.w_25_4 ctx) (parallel_generatedRoot_273.w_25_5 ctx) outputs) :
    parallel_generatedRoot_273.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_273.Facts  (ctx : parallel_generatedRoot_273Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_273.Facts_0 ctx outputs

theorem parallel_generatedRoot_273.facts  (ctx : parallel_generatedRoot_273Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_273.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_273.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_273.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_277.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_277.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_277Ctx" => (parallel_generatedRoot_277.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_277.w_0_0  (ctx : parallel_generatedRoot_277Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_277.w_1_0  (ctx : parallel_generatedRoot_277Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_277.w_2_0  (ctx : parallel_generatedRoot_277Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_277.w_3_0  (ctx : parallel_generatedRoot_277Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_277.w_4_0  (ctx : parallel_generatedRoot_277Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_277.w_5_0  (ctx : parallel_generatedRoot_277Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_277.w_6_0  (ctx : parallel_generatedRoot_277Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_277.w_6_1  (ctx : parallel_generatedRoot_277Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_277.w_6_2  (ctx : parallel_generatedRoot_277Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_277.w_6_3  (ctx : parallel_generatedRoot_277Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_277.w_6_4  (ctx : parallel_generatedRoot_277Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_277.w_6_5  (ctx : parallel_generatedRoot_277Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_277.w_7_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_277.w_8_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_277.w_9_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_277.w_7_0 ctx) (parallel_generatedRoot_277.w_8_0 ctx)

def parallel_generatedRoot_277.w_10_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_277.w_11_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_277.w_12_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_277.w_10_0 ctx) (parallel_generatedRoot_277.w_11_0 ctx)

def parallel_generatedRoot_277.w_13_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_277.w_14_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_277.w_15_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_277.w_13_0 ctx) (parallel_generatedRoot_277.w_14_0 ctx)

def parallel_generatedRoot_277.w_16_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_277.w_17_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_277.w_18_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_277.w_16_0 ctx) (parallel_generatedRoot_277.w_17_0 ctx)

def parallel_generatedRoot_277.w_19_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_277.w_20_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_277.w_21_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_277.w_19_0 ctx) (parallel_generatedRoot_277.w_20_0 ctx)

def parallel_generatedRoot_277.w_22_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_277.w_23_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_277.w_24_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_277.w_22_0 ctx) (parallel_generatedRoot_277.w_23_0 ctx)

def parallel_generatedRoot_277.w_25_0  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_277.w_25_1  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_277.w_25_2  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_277.w_25_3  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_277.w_25_4  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_277.w_25_5  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_277.outs  (ctx : parallel_generatedRoot_277Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_277.w_25_0 ctx), ((parallel_generatedRoot_277.w_25_1 ctx), ((parallel_generatedRoot_277.w_25_2 ctx), ((parallel_generatedRoot_277.w_25_3 ctx), ((parallel_generatedRoot_277.w_25_4 ctx), ((parallel_generatedRoot_277.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_277.Facts_0  (ctx : parallel_generatedRoot_277Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_277_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_277.w_0_0 ctx), ((parallel_generatedRoot_277.w_1_0 ctx), ((parallel_generatedRoot_277.w_2_0 ctx), ((parallel_generatedRoot_277.w_3_0 ctx), ((parallel_generatedRoot_277.w_4_0 ctx), ((parallel_generatedRoot_277.w_5_0 ctx), ())))))) (((parallel_generatedRoot_277.w_6_0 ctx) i), (((parallel_generatedRoot_277.w_6_1 ctx) i), (((parallel_generatedRoot_277.w_6_2 ctx) i), (((parallel_generatedRoot_277.w_6_3 ctx) i), (((parallel_generatedRoot_277.w_6_4 ctx) i), (((parallel_generatedRoot_277.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_0 ctx) (0) (parallel_generatedRoot_277.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_0 ctx) (1) (parallel_generatedRoot_277.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_1 ctx) (0) (parallel_generatedRoot_277.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_1 ctx) (1) (parallel_generatedRoot_277.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_2 ctx) (0) (parallel_generatedRoot_277.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_2 ctx) (1) (parallel_generatedRoot_277.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_3 ctx) (0) (parallel_generatedRoot_277.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_3 ctx) (1) (parallel_generatedRoot_277.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_4 ctx) (0) (parallel_generatedRoot_277.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_4 ctx) (1) (parallel_generatedRoot_277.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_5 ctx) (0) (parallel_generatedRoot_277.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_277.w_6_5 ctx) (1) (parallel_generatedRoot_277.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_277_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_277.w_6_0 ctx), ((parallel_generatedRoot_277.w_6_1 ctx), ((parallel_generatedRoot_277.w_6_2 ctx), ((parallel_generatedRoot_277.w_6_3 ctx), ((parallel_generatedRoot_277.w_6_4 ctx), ((parallel_generatedRoot_277.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_277.w_9_0 ctx), ((parallel_generatedRoot_277.w_12_0 ctx), ((parallel_generatedRoot_277.w_15_0 ctx), ((parallel_generatedRoot_277.w_18_0 ctx), ((parallel_generatedRoot_277.w_21_0 ctx), ((parallel_generatedRoot_277.w_24_0 ctx), ())))))) ((parallel_generatedRoot_277.w_25_0 ctx), ((parallel_generatedRoot_277.w_25_1 ctx), ((parallel_generatedRoot_277.w_25_2 ctx), ((parallel_generatedRoot_277.w_25_3 ctx), ((parallel_generatedRoot_277.w_25_4 ctx), ((parallel_generatedRoot_277.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_277.outs ctx

theorem parallel_generatedRoot_277.facts_0  {ctx : parallel_generatedRoot_277Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_277.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_277.w_0_0 ctx) (parallel_generatedRoot_277.w_1_0 ctx) (parallel_generatedRoot_277.w_2_0 ctx) (parallel_generatedRoot_277.w_3_0 ctx) (parallel_generatedRoot_277.w_4_0 ctx) (parallel_generatedRoot_277.w_5_0 ctx) (parallel_generatedRoot_277.w_6_0 ctx) (parallel_generatedRoot_277.w_6_1 ctx) (parallel_generatedRoot_277.w_6_2 ctx) (parallel_generatedRoot_277.w_6_3 ctx) (parallel_generatedRoot_277.w_6_4 ctx) (parallel_generatedRoot_277.w_6_5 ctx) (parallel_generatedRoot_277.w_7_0 ctx) (parallel_generatedRoot_277.w_8_0 ctx) (parallel_generatedRoot_277.w_9_0 ctx) (parallel_generatedRoot_277.w_10_0 ctx) (parallel_generatedRoot_277.w_11_0 ctx) (parallel_generatedRoot_277.w_12_0 ctx) (parallel_generatedRoot_277.w_13_0 ctx) (parallel_generatedRoot_277.w_14_0 ctx) (parallel_generatedRoot_277.w_15_0 ctx) (parallel_generatedRoot_277.w_16_0 ctx) (parallel_generatedRoot_277.w_17_0 ctx) (parallel_generatedRoot_277.w_18_0 ctx) (parallel_generatedRoot_277.w_19_0 ctx) (parallel_generatedRoot_277.w_20_0 ctx) (parallel_generatedRoot_277.w_21_0 ctx) (parallel_generatedRoot_277.w_22_0 ctx) (parallel_generatedRoot_277.w_23_0 ctx) (parallel_generatedRoot_277.w_24_0 ctx) (parallel_generatedRoot_277.w_25_0 ctx) (parallel_generatedRoot_277.w_25_1 ctx) (parallel_generatedRoot_277.w_25_2 ctx) (parallel_generatedRoot_277.w_25_3 ctx) (parallel_generatedRoot_277.w_25_4 ctx) (parallel_generatedRoot_277.w_25_5 ctx) outputs) :
    parallel_generatedRoot_277.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_277.Facts  (ctx : parallel_generatedRoot_277Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_277.Facts_0 ctx outputs

theorem parallel_generatedRoot_277.facts  (ctx : parallel_generatedRoot_277Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_277.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_277.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_277.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_281.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_281.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_281Ctx" => (parallel_generatedRoot_281.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_281.w_0_0  (ctx : parallel_generatedRoot_281Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_281.w_1_0  (ctx : parallel_generatedRoot_281Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_281.w_2_0  (ctx : parallel_generatedRoot_281Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_281.w_3_0  (ctx : parallel_generatedRoot_281Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_281.w_4_0  (ctx : parallel_generatedRoot_281Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_281.w_5_0  (ctx : parallel_generatedRoot_281Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_281.w_6_0  (ctx : parallel_generatedRoot_281Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_281.w_6_1  (ctx : parallel_generatedRoot_281Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_281.w_6_2  (ctx : parallel_generatedRoot_281Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_281.w_6_3  (ctx : parallel_generatedRoot_281Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_281.w_6_4  (ctx : parallel_generatedRoot_281Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_281.w_6_5  (ctx : parallel_generatedRoot_281Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_281.w_7_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_281.w_8_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_281.w_9_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_281.w_7_0 ctx) (parallel_generatedRoot_281.w_8_0 ctx)

def parallel_generatedRoot_281.w_10_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_281.w_11_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_281.w_12_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_281.w_10_0 ctx) (parallel_generatedRoot_281.w_11_0 ctx)

def parallel_generatedRoot_281.w_13_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_281.w_14_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_281.w_15_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_281.w_13_0 ctx) (parallel_generatedRoot_281.w_14_0 ctx)

def parallel_generatedRoot_281.w_16_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_281.w_17_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_281.w_18_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_281.w_16_0 ctx) (parallel_generatedRoot_281.w_17_0 ctx)

def parallel_generatedRoot_281.w_19_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_281.w_20_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_281.w_21_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_281.w_19_0 ctx) (parallel_generatedRoot_281.w_20_0 ctx)

def parallel_generatedRoot_281.w_22_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_281.w_23_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_281.w_24_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_281.w_22_0 ctx) (parallel_generatedRoot_281.w_23_0 ctx)

def parallel_generatedRoot_281.w_25_0  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_281.w_25_1  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_281.w_25_2  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_281.w_25_3  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_281.w_25_4  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_281.w_25_5  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_281.outs  (ctx : parallel_generatedRoot_281Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_281.w_25_0 ctx), ((parallel_generatedRoot_281.w_25_1 ctx), ((parallel_generatedRoot_281.w_25_2 ctx), ((parallel_generatedRoot_281.w_25_3 ctx), ((parallel_generatedRoot_281.w_25_4 ctx), ((parallel_generatedRoot_281.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_281.Facts_0  (ctx : parallel_generatedRoot_281Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_281_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_281.w_0_0 ctx), ((parallel_generatedRoot_281.w_1_0 ctx), ((parallel_generatedRoot_281.w_2_0 ctx), ((parallel_generatedRoot_281.w_3_0 ctx), ((parallel_generatedRoot_281.w_4_0 ctx), ((parallel_generatedRoot_281.w_5_0 ctx), ())))))) (((parallel_generatedRoot_281.w_6_0 ctx) i), (((parallel_generatedRoot_281.w_6_1 ctx) i), (((parallel_generatedRoot_281.w_6_2 ctx) i), (((parallel_generatedRoot_281.w_6_3 ctx) i), (((parallel_generatedRoot_281.w_6_4 ctx) i), (((parallel_generatedRoot_281.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_0 ctx) (0) (parallel_generatedRoot_281.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_0 ctx) (1) (parallel_generatedRoot_281.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_1 ctx) (0) (parallel_generatedRoot_281.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_1 ctx) (1) (parallel_generatedRoot_281.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_2 ctx) (0) (parallel_generatedRoot_281.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_2 ctx) (1) (parallel_generatedRoot_281.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_3 ctx) (0) (parallel_generatedRoot_281.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_3 ctx) (1) (parallel_generatedRoot_281.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_4 ctx) (0) (parallel_generatedRoot_281.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_4 ctx) (1) (parallel_generatedRoot_281.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_5 ctx) (0) (parallel_generatedRoot_281.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_281.w_6_5 ctx) (1) (parallel_generatedRoot_281.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_281_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_281.w_6_0 ctx), ((parallel_generatedRoot_281.w_6_1 ctx), ((parallel_generatedRoot_281.w_6_2 ctx), ((parallel_generatedRoot_281.w_6_3 ctx), ((parallel_generatedRoot_281.w_6_4 ctx), ((parallel_generatedRoot_281.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_281.w_9_0 ctx), ((parallel_generatedRoot_281.w_12_0 ctx), ((parallel_generatedRoot_281.w_15_0 ctx), ((parallel_generatedRoot_281.w_18_0 ctx), ((parallel_generatedRoot_281.w_21_0 ctx), ((parallel_generatedRoot_281.w_24_0 ctx), ())))))) ((parallel_generatedRoot_281.w_25_0 ctx), ((parallel_generatedRoot_281.w_25_1 ctx), ((parallel_generatedRoot_281.w_25_2 ctx), ((parallel_generatedRoot_281.w_25_3 ctx), ((parallel_generatedRoot_281.w_25_4 ctx), ((parallel_generatedRoot_281.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_281.outs ctx

theorem parallel_generatedRoot_281.facts_0  {ctx : parallel_generatedRoot_281Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_281.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_281.w_0_0 ctx) (parallel_generatedRoot_281.w_1_0 ctx) (parallel_generatedRoot_281.w_2_0 ctx) (parallel_generatedRoot_281.w_3_0 ctx) (parallel_generatedRoot_281.w_4_0 ctx) (parallel_generatedRoot_281.w_5_0 ctx) (parallel_generatedRoot_281.w_6_0 ctx) (parallel_generatedRoot_281.w_6_1 ctx) (parallel_generatedRoot_281.w_6_2 ctx) (parallel_generatedRoot_281.w_6_3 ctx) (parallel_generatedRoot_281.w_6_4 ctx) (parallel_generatedRoot_281.w_6_5 ctx) (parallel_generatedRoot_281.w_7_0 ctx) (parallel_generatedRoot_281.w_8_0 ctx) (parallel_generatedRoot_281.w_9_0 ctx) (parallel_generatedRoot_281.w_10_0 ctx) (parallel_generatedRoot_281.w_11_0 ctx) (parallel_generatedRoot_281.w_12_0 ctx) (parallel_generatedRoot_281.w_13_0 ctx) (parallel_generatedRoot_281.w_14_0 ctx) (parallel_generatedRoot_281.w_15_0 ctx) (parallel_generatedRoot_281.w_16_0 ctx) (parallel_generatedRoot_281.w_17_0 ctx) (parallel_generatedRoot_281.w_18_0 ctx) (parallel_generatedRoot_281.w_19_0 ctx) (parallel_generatedRoot_281.w_20_0 ctx) (parallel_generatedRoot_281.w_21_0 ctx) (parallel_generatedRoot_281.w_22_0 ctx) (parallel_generatedRoot_281.w_23_0 ctx) (parallel_generatedRoot_281.w_24_0 ctx) (parallel_generatedRoot_281.w_25_0 ctx) (parallel_generatedRoot_281.w_25_1 ctx) (parallel_generatedRoot_281.w_25_2 ctx) (parallel_generatedRoot_281.w_25_3 ctx) (parallel_generatedRoot_281.w_25_4 ctx) (parallel_generatedRoot_281.w_25_5 ctx) outputs) :
    parallel_generatedRoot_281.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_281.Facts  (ctx : parallel_generatedRoot_281Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_281.Facts_0 ctx outputs

theorem parallel_generatedRoot_281.facts  (ctx : parallel_generatedRoot_281Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_281.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_281.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_281.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_285.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_285.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_285Ctx" => (parallel_generatedRoot_285.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_285.w_0_0  (ctx : parallel_generatedRoot_285Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_285.w_1_0  (ctx : parallel_generatedRoot_285Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_285.w_2_0  (ctx : parallel_generatedRoot_285Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_285.w_3_0  (ctx : parallel_generatedRoot_285Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_285.w_4_0  (ctx : parallel_generatedRoot_285Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_285.w_5_0  (ctx : parallel_generatedRoot_285Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_285.w_6_0  (ctx : parallel_generatedRoot_285Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_285.w_6_1  (ctx : parallel_generatedRoot_285Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_285.w_6_2  (ctx : parallel_generatedRoot_285Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_285.w_6_3  (ctx : parallel_generatedRoot_285Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_285.w_6_4  (ctx : parallel_generatedRoot_285Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_285.w_6_5  (ctx : parallel_generatedRoot_285Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_285.w_7_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_285.w_8_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_285.w_9_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_285.w_7_0 ctx) (parallel_generatedRoot_285.w_8_0 ctx)

def parallel_generatedRoot_285.w_10_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_285.w_11_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_285.w_12_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_285.w_10_0 ctx) (parallel_generatedRoot_285.w_11_0 ctx)

def parallel_generatedRoot_285.w_13_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_285.w_14_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_285.w_15_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_285.w_13_0 ctx) (parallel_generatedRoot_285.w_14_0 ctx)

def parallel_generatedRoot_285.w_16_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_285.w_17_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_285.w_18_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_285.w_16_0 ctx) (parallel_generatedRoot_285.w_17_0 ctx)

def parallel_generatedRoot_285.w_19_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_285.w_20_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_285.w_21_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_285.w_19_0 ctx) (parallel_generatedRoot_285.w_20_0 ctx)

def parallel_generatedRoot_285.w_22_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_285.w_23_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_285.w_24_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_285.w_22_0 ctx) (parallel_generatedRoot_285.w_23_0 ctx)

def parallel_generatedRoot_285.w_25_0  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_285.w_25_1  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_285.w_25_2  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_285.w_25_3  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_285.w_25_4  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_285.w_25_5  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_285.outs  (ctx : parallel_generatedRoot_285Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_285.w_25_0 ctx), ((parallel_generatedRoot_285.w_25_1 ctx), ((parallel_generatedRoot_285.w_25_2 ctx), ((parallel_generatedRoot_285.w_25_3 ctx), ((parallel_generatedRoot_285.w_25_4 ctx), ((parallel_generatedRoot_285.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_285.Facts_0  (ctx : parallel_generatedRoot_285Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_285_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_285.w_0_0 ctx), ((parallel_generatedRoot_285.w_1_0 ctx), ((parallel_generatedRoot_285.w_2_0 ctx), ((parallel_generatedRoot_285.w_3_0 ctx), ((parallel_generatedRoot_285.w_4_0 ctx), ((parallel_generatedRoot_285.w_5_0 ctx), ())))))) (((parallel_generatedRoot_285.w_6_0 ctx) i), (((parallel_generatedRoot_285.w_6_1 ctx) i), (((parallel_generatedRoot_285.w_6_2 ctx) i), (((parallel_generatedRoot_285.w_6_3 ctx) i), (((parallel_generatedRoot_285.w_6_4 ctx) i), (((parallel_generatedRoot_285.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_0 ctx) (0) (parallel_generatedRoot_285.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_0 ctx) (1) (parallel_generatedRoot_285.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_1 ctx) (0) (parallel_generatedRoot_285.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_1 ctx) (1) (parallel_generatedRoot_285.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_2 ctx) (0) (parallel_generatedRoot_285.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_2 ctx) (1) (parallel_generatedRoot_285.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_3 ctx) (0) (parallel_generatedRoot_285.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_3 ctx) (1) (parallel_generatedRoot_285.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_4 ctx) (0) (parallel_generatedRoot_285.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_4 ctx) (1) (parallel_generatedRoot_285.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_5 ctx) (0) (parallel_generatedRoot_285.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_285.w_6_5 ctx) (1) (parallel_generatedRoot_285.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_285_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_285.w_6_0 ctx), ((parallel_generatedRoot_285.w_6_1 ctx), ((parallel_generatedRoot_285.w_6_2 ctx), ((parallel_generatedRoot_285.w_6_3 ctx), ((parallel_generatedRoot_285.w_6_4 ctx), ((parallel_generatedRoot_285.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_285.w_9_0 ctx), ((parallel_generatedRoot_285.w_12_0 ctx), ((parallel_generatedRoot_285.w_15_0 ctx), ((parallel_generatedRoot_285.w_18_0 ctx), ((parallel_generatedRoot_285.w_21_0 ctx), ((parallel_generatedRoot_285.w_24_0 ctx), ())))))) ((parallel_generatedRoot_285.w_25_0 ctx), ((parallel_generatedRoot_285.w_25_1 ctx), ((parallel_generatedRoot_285.w_25_2 ctx), ((parallel_generatedRoot_285.w_25_3 ctx), ((parallel_generatedRoot_285.w_25_4 ctx), ((parallel_generatedRoot_285.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_285.outs ctx

theorem parallel_generatedRoot_285.facts_0  {ctx : parallel_generatedRoot_285Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_285.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_285.w_0_0 ctx) (parallel_generatedRoot_285.w_1_0 ctx) (parallel_generatedRoot_285.w_2_0 ctx) (parallel_generatedRoot_285.w_3_0 ctx) (parallel_generatedRoot_285.w_4_0 ctx) (parallel_generatedRoot_285.w_5_0 ctx) (parallel_generatedRoot_285.w_6_0 ctx) (parallel_generatedRoot_285.w_6_1 ctx) (parallel_generatedRoot_285.w_6_2 ctx) (parallel_generatedRoot_285.w_6_3 ctx) (parallel_generatedRoot_285.w_6_4 ctx) (parallel_generatedRoot_285.w_6_5 ctx) (parallel_generatedRoot_285.w_7_0 ctx) (parallel_generatedRoot_285.w_8_0 ctx) (parallel_generatedRoot_285.w_9_0 ctx) (parallel_generatedRoot_285.w_10_0 ctx) (parallel_generatedRoot_285.w_11_0 ctx) (parallel_generatedRoot_285.w_12_0 ctx) (parallel_generatedRoot_285.w_13_0 ctx) (parallel_generatedRoot_285.w_14_0 ctx) (parallel_generatedRoot_285.w_15_0 ctx) (parallel_generatedRoot_285.w_16_0 ctx) (parallel_generatedRoot_285.w_17_0 ctx) (parallel_generatedRoot_285.w_18_0 ctx) (parallel_generatedRoot_285.w_19_0 ctx) (parallel_generatedRoot_285.w_20_0 ctx) (parallel_generatedRoot_285.w_21_0 ctx) (parallel_generatedRoot_285.w_22_0 ctx) (parallel_generatedRoot_285.w_23_0 ctx) (parallel_generatedRoot_285.w_24_0 ctx) (parallel_generatedRoot_285.w_25_0 ctx) (parallel_generatedRoot_285.w_25_1 ctx) (parallel_generatedRoot_285.w_25_2 ctx) (parallel_generatedRoot_285.w_25_3 ctx) (parallel_generatedRoot_285.w_25_4 ctx) (parallel_generatedRoot_285.w_25_5 ctx) outputs) :
    parallel_generatedRoot_285.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_285.Facts  (ctx : parallel_generatedRoot_285Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_285.Facts_0 ctx outputs

theorem parallel_generatedRoot_285.facts  (ctx : parallel_generatedRoot_285Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_285.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_285.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_285.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_289.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_289.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_289Ctx" => (parallel_generatedRoot_289.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_289.w_0_0  (ctx : parallel_generatedRoot_289Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_289.w_1_0  (ctx : parallel_generatedRoot_289Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_289.w_2_0  (ctx : parallel_generatedRoot_289Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_289.w_3_0  (ctx : parallel_generatedRoot_289Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_289.w_4_0  (ctx : parallel_generatedRoot_289Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_289.w_5_0  (ctx : parallel_generatedRoot_289Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_289.w_6_0  (ctx : parallel_generatedRoot_289Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_289.w_6_1  (ctx : parallel_generatedRoot_289Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_289.w_6_2  (ctx : parallel_generatedRoot_289Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_289.w_6_3  (ctx : parallel_generatedRoot_289Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_289.w_6_4  (ctx : parallel_generatedRoot_289Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_289.w_6_5  (ctx : parallel_generatedRoot_289Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_289.w_7_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_289.w_8_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_289.w_9_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_289.w_7_0 ctx) (parallel_generatedRoot_289.w_8_0 ctx)

def parallel_generatedRoot_289.w_10_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_289.w_11_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_289.w_12_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_289.w_10_0 ctx) (parallel_generatedRoot_289.w_11_0 ctx)

def parallel_generatedRoot_289.w_13_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_289.w_14_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_289.w_15_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_289.w_13_0 ctx) (parallel_generatedRoot_289.w_14_0 ctx)

def parallel_generatedRoot_289.w_16_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_289.w_17_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_289.w_18_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_289.w_16_0 ctx) (parallel_generatedRoot_289.w_17_0 ctx)

def parallel_generatedRoot_289.w_19_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_289.w_20_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_289.w_21_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_289.w_19_0 ctx) (parallel_generatedRoot_289.w_20_0 ctx)

def parallel_generatedRoot_289.w_22_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_289.w_23_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_289.w_24_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_289.w_22_0 ctx) (parallel_generatedRoot_289.w_23_0 ctx)

def parallel_generatedRoot_289.w_25_0  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_289.w_25_1  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_289.w_25_2  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_289.w_25_3  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_289.w_25_4  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_289.w_25_5  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_289.outs  (ctx : parallel_generatedRoot_289Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_289.w_25_0 ctx), ((parallel_generatedRoot_289.w_25_1 ctx), ((parallel_generatedRoot_289.w_25_2 ctx), ((parallel_generatedRoot_289.w_25_3 ctx), ((parallel_generatedRoot_289.w_25_4 ctx), ((parallel_generatedRoot_289.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_289.Facts_0  (ctx : parallel_generatedRoot_289Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_289_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_289.w_0_0 ctx), ((parallel_generatedRoot_289.w_1_0 ctx), ((parallel_generatedRoot_289.w_2_0 ctx), ((parallel_generatedRoot_289.w_3_0 ctx), ((parallel_generatedRoot_289.w_4_0 ctx), ((parallel_generatedRoot_289.w_5_0 ctx), ())))))) (((parallel_generatedRoot_289.w_6_0 ctx) i), (((parallel_generatedRoot_289.w_6_1 ctx) i), (((parallel_generatedRoot_289.w_6_2 ctx) i), (((parallel_generatedRoot_289.w_6_3 ctx) i), (((parallel_generatedRoot_289.w_6_4 ctx) i), (((parallel_generatedRoot_289.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_0 ctx) (0) (parallel_generatedRoot_289.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_0 ctx) (1) (parallel_generatedRoot_289.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_1 ctx) (0) (parallel_generatedRoot_289.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_1 ctx) (1) (parallel_generatedRoot_289.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_2 ctx) (0) (parallel_generatedRoot_289.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_2 ctx) (1) (parallel_generatedRoot_289.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_3 ctx) (0) (parallel_generatedRoot_289.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_3 ctx) (1) (parallel_generatedRoot_289.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_4 ctx) (0) (parallel_generatedRoot_289.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_4 ctx) (1) (parallel_generatedRoot_289.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_5 ctx) (0) (parallel_generatedRoot_289.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_289.w_6_5 ctx) (1) (parallel_generatedRoot_289.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_289_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_289.w_6_0 ctx), ((parallel_generatedRoot_289.w_6_1 ctx), ((parallel_generatedRoot_289.w_6_2 ctx), ((parallel_generatedRoot_289.w_6_3 ctx), ((parallel_generatedRoot_289.w_6_4 ctx), ((parallel_generatedRoot_289.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_289.w_9_0 ctx), ((parallel_generatedRoot_289.w_12_0 ctx), ((parallel_generatedRoot_289.w_15_0 ctx), ((parallel_generatedRoot_289.w_18_0 ctx), ((parallel_generatedRoot_289.w_21_0 ctx), ((parallel_generatedRoot_289.w_24_0 ctx), ())))))) ((parallel_generatedRoot_289.w_25_0 ctx), ((parallel_generatedRoot_289.w_25_1 ctx), ((parallel_generatedRoot_289.w_25_2 ctx), ((parallel_generatedRoot_289.w_25_3 ctx), ((parallel_generatedRoot_289.w_25_4 ctx), ((parallel_generatedRoot_289.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_289.outs ctx

theorem parallel_generatedRoot_289.facts_0  {ctx : parallel_generatedRoot_289Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_289.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_289.w_0_0 ctx) (parallel_generatedRoot_289.w_1_0 ctx) (parallel_generatedRoot_289.w_2_0 ctx) (parallel_generatedRoot_289.w_3_0 ctx) (parallel_generatedRoot_289.w_4_0 ctx) (parallel_generatedRoot_289.w_5_0 ctx) (parallel_generatedRoot_289.w_6_0 ctx) (parallel_generatedRoot_289.w_6_1 ctx) (parallel_generatedRoot_289.w_6_2 ctx) (parallel_generatedRoot_289.w_6_3 ctx) (parallel_generatedRoot_289.w_6_4 ctx) (parallel_generatedRoot_289.w_6_5 ctx) (parallel_generatedRoot_289.w_7_0 ctx) (parallel_generatedRoot_289.w_8_0 ctx) (parallel_generatedRoot_289.w_9_0 ctx) (parallel_generatedRoot_289.w_10_0 ctx) (parallel_generatedRoot_289.w_11_0 ctx) (parallel_generatedRoot_289.w_12_0 ctx) (parallel_generatedRoot_289.w_13_0 ctx) (parallel_generatedRoot_289.w_14_0 ctx) (parallel_generatedRoot_289.w_15_0 ctx) (parallel_generatedRoot_289.w_16_0 ctx) (parallel_generatedRoot_289.w_17_0 ctx) (parallel_generatedRoot_289.w_18_0 ctx) (parallel_generatedRoot_289.w_19_0 ctx) (parallel_generatedRoot_289.w_20_0 ctx) (parallel_generatedRoot_289.w_21_0 ctx) (parallel_generatedRoot_289.w_22_0 ctx) (parallel_generatedRoot_289.w_23_0 ctx) (parallel_generatedRoot_289.w_24_0 ctx) (parallel_generatedRoot_289.w_25_0 ctx) (parallel_generatedRoot_289.w_25_1 ctx) (parallel_generatedRoot_289.w_25_2 ctx) (parallel_generatedRoot_289.w_25_3 ctx) (parallel_generatedRoot_289.w_25_4 ctx) (parallel_generatedRoot_289.w_25_5 ctx) outputs) :
    parallel_generatedRoot_289.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_289.Facts  (ctx : parallel_generatedRoot_289Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_289.Facts_0 ctx outputs

theorem parallel_generatedRoot_289.facts  (ctx : parallel_generatedRoot_289Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_289.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_289.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_289.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_293.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_293.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_293Ctx" => (parallel_generatedRoot_293.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_293.w_0_0  (ctx : parallel_generatedRoot_293Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_293.w_1_0  (ctx : parallel_generatedRoot_293Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_293.w_2_0  (ctx : parallel_generatedRoot_293Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_293.w_3_0  (ctx : parallel_generatedRoot_293Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_293.w_4_0  (ctx : parallel_generatedRoot_293Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_293.w_5_0  (ctx : parallel_generatedRoot_293Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_293.w_6_0  (ctx : parallel_generatedRoot_293Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_293.w_6_1  (ctx : parallel_generatedRoot_293Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_293.w_6_2  (ctx : parallel_generatedRoot_293Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_293.w_6_3  (ctx : parallel_generatedRoot_293Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_293.w_6_4  (ctx : parallel_generatedRoot_293Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_293.w_6_5  (ctx : parallel_generatedRoot_293Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_293.w_7_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_293.w_8_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_293.w_9_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_293.w_7_0 ctx) (parallel_generatedRoot_293.w_8_0 ctx)

def parallel_generatedRoot_293.w_10_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_293.w_11_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_293.w_12_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_293.w_10_0 ctx) (parallel_generatedRoot_293.w_11_0 ctx)

def parallel_generatedRoot_293.w_13_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_293.w_14_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_293.w_15_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_293.w_13_0 ctx) (parallel_generatedRoot_293.w_14_0 ctx)

def parallel_generatedRoot_293.w_16_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_293.w_17_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_293.w_18_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_293.w_16_0 ctx) (parallel_generatedRoot_293.w_17_0 ctx)

def parallel_generatedRoot_293.w_19_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_293.w_20_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_293.w_21_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_293.w_19_0 ctx) (parallel_generatedRoot_293.w_20_0 ctx)

def parallel_generatedRoot_293.w_22_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_293.w_23_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_293.w_24_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_293.w_22_0 ctx) (parallel_generatedRoot_293.w_23_0 ctx)

def parallel_generatedRoot_293.w_25_0  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_293.w_25_1  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_293.w_25_2  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_293.w_25_3  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_293.w_25_4  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_293.w_25_5  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_293.outs  (ctx : parallel_generatedRoot_293Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_293.w_25_0 ctx), ((parallel_generatedRoot_293.w_25_1 ctx), ((parallel_generatedRoot_293.w_25_2 ctx), ((parallel_generatedRoot_293.w_25_3 ctx), ((parallel_generatedRoot_293.w_25_4 ctx), ((parallel_generatedRoot_293.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_293.Facts_0  (ctx : parallel_generatedRoot_293Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_293_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_293.w_0_0 ctx), ((parallel_generatedRoot_293.w_1_0 ctx), ((parallel_generatedRoot_293.w_2_0 ctx), ((parallel_generatedRoot_293.w_3_0 ctx), ((parallel_generatedRoot_293.w_4_0 ctx), ((parallel_generatedRoot_293.w_5_0 ctx), ())))))) (((parallel_generatedRoot_293.w_6_0 ctx) i), (((parallel_generatedRoot_293.w_6_1 ctx) i), (((parallel_generatedRoot_293.w_6_2 ctx) i), (((parallel_generatedRoot_293.w_6_3 ctx) i), (((parallel_generatedRoot_293.w_6_4 ctx) i), (((parallel_generatedRoot_293.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_0 ctx) (0) (parallel_generatedRoot_293.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_0 ctx) (1) (parallel_generatedRoot_293.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_1 ctx) (0) (parallel_generatedRoot_293.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_1 ctx) (1) (parallel_generatedRoot_293.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_2 ctx) (0) (parallel_generatedRoot_293.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_2 ctx) (1) (parallel_generatedRoot_293.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_3 ctx) (0) (parallel_generatedRoot_293.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_3 ctx) (1) (parallel_generatedRoot_293.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_4 ctx) (0) (parallel_generatedRoot_293.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_4 ctx) (1) (parallel_generatedRoot_293.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_5 ctx) (0) (parallel_generatedRoot_293.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_293.w_6_5 ctx) (1) (parallel_generatedRoot_293.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_293_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_293.w_6_0 ctx), ((parallel_generatedRoot_293.w_6_1 ctx), ((parallel_generatedRoot_293.w_6_2 ctx), ((parallel_generatedRoot_293.w_6_3 ctx), ((parallel_generatedRoot_293.w_6_4 ctx), ((parallel_generatedRoot_293.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_293.w_9_0 ctx), ((parallel_generatedRoot_293.w_12_0 ctx), ((parallel_generatedRoot_293.w_15_0 ctx), ((parallel_generatedRoot_293.w_18_0 ctx), ((parallel_generatedRoot_293.w_21_0 ctx), ((parallel_generatedRoot_293.w_24_0 ctx), ())))))) ((parallel_generatedRoot_293.w_25_0 ctx), ((parallel_generatedRoot_293.w_25_1 ctx), ((parallel_generatedRoot_293.w_25_2 ctx), ((parallel_generatedRoot_293.w_25_3 ctx), ((parallel_generatedRoot_293.w_25_4 ctx), ((parallel_generatedRoot_293.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_293.outs ctx

theorem parallel_generatedRoot_293.facts_0  {ctx : parallel_generatedRoot_293Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_293.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_293.w_0_0 ctx) (parallel_generatedRoot_293.w_1_0 ctx) (parallel_generatedRoot_293.w_2_0 ctx) (parallel_generatedRoot_293.w_3_0 ctx) (parallel_generatedRoot_293.w_4_0 ctx) (parallel_generatedRoot_293.w_5_0 ctx) (parallel_generatedRoot_293.w_6_0 ctx) (parallel_generatedRoot_293.w_6_1 ctx) (parallel_generatedRoot_293.w_6_2 ctx) (parallel_generatedRoot_293.w_6_3 ctx) (parallel_generatedRoot_293.w_6_4 ctx) (parallel_generatedRoot_293.w_6_5 ctx) (parallel_generatedRoot_293.w_7_0 ctx) (parallel_generatedRoot_293.w_8_0 ctx) (parallel_generatedRoot_293.w_9_0 ctx) (parallel_generatedRoot_293.w_10_0 ctx) (parallel_generatedRoot_293.w_11_0 ctx) (parallel_generatedRoot_293.w_12_0 ctx) (parallel_generatedRoot_293.w_13_0 ctx) (parallel_generatedRoot_293.w_14_0 ctx) (parallel_generatedRoot_293.w_15_0 ctx) (parallel_generatedRoot_293.w_16_0 ctx) (parallel_generatedRoot_293.w_17_0 ctx) (parallel_generatedRoot_293.w_18_0 ctx) (parallel_generatedRoot_293.w_19_0 ctx) (parallel_generatedRoot_293.w_20_0 ctx) (parallel_generatedRoot_293.w_21_0 ctx) (parallel_generatedRoot_293.w_22_0 ctx) (parallel_generatedRoot_293.w_23_0 ctx) (parallel_generatedRoot_293.w_24_0 ctx) (parallel_generatedRoot_293.w_25_0 ctx) (parallel_generatedRoot_293.w_25_1 ctx) (parallel_generatedRoot_293.w_25_2 ctx) (parallel_generatedRoot_293.w_25_3 ctx) (parallel_generatedRoot_293.w_25_4 ctx) (parallel_generatedRoot_293.w_25_5 ctx) outputs) :
    parallel_generatedRoot_293.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_293.Facts  (ctx : parallel_generatedRoot_293Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_293.Facts_0 ctx outputs

theorem parallel_generatedRoot_293.facts  (ctx : parallel_generatedRoot_293Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_293.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_293.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_293.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_297.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_297.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_297Ctx" => (parallel_generatedRoot_297.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_297.w_0_0  (ctx : parallel_generatedRoot_297Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_297.w_1_0  (ctx : parallel_generatedRoot_297Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_297.w_2_0  (ctx : parallel_generatedRoot_297Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_297.w_3_0  (ctx : parallel_generatedRoot_297Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_297.w_4_0  (ctx : parallel_generatedRoot_297Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_297.w_5_0  (ctx : parallel_generatedRoot_297Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_297.w_6_0  (ctx : parallel_generatedRoot_297Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_297.w_6_1  (ctx : parallel_generatedRoot_297Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_297.w_6_2  (ctx : parallel_generatedRoot_297Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_297.w_6_3  (ctx : parallel_generatedRoot_297Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_297.w_6_4  (ctx : parallel_generatedRoot_297Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_297.w_6_5  (ctx : parallel_generatedRoot_297Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_297.w_7_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_297.w_8_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_297.w_9_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_297.w_7_0 ctx) (parallel_generatedRoot_297.w_8_0 ctx)

def parallel_generatedRoot_297.w_10_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_297.w_11_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_297.w_12_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_297.w_10_0 ctx) (parallel_generatedRoot_297.w_11_0 ctx)

def parallel_generatedRoot_297.w_13_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_297.w_14_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_297.w_15_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_297.w_13_0 ctx) (parallel_generatedRoot_297.w_14_0 ctx)

def parallel_generatedRoot_297.w_16_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_297.w_17_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_297.w_18_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_297.w_16_0 ctx) (parallel_generatedRoot_297.w_17_0 ctx)

def parallel_generatedRoot_297.w_19_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_297.w_20_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_297.w_21_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_297.w_19_0 ctx) (parallel_generatedRoot_297.w_20_0 ctx)

def parallel_generatedRoot_297.w_22_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_297.w_23_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_297.w_24_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_297.w_22_0 ctx) (parallel_generatedRoot_297.w_23_0 ctx)

def parallel_generatedRoot_297.w_25_0  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_297.w_25_1  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_297.w_25_2  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_297.w_25_3  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_297.w_25_4  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_297.w_25_5  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_297.outs  (ctx : parallel_generatedRoot_297Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_297.w_25_0 ctx), ((parallel_generatedRoot_297.w_25_1 ctx), ((parallel_generatedRoot_297.w_25_2 ctx), ((parallel_generatedRoot_297.w_25_3 ctx), ((parallel_generatedRoot_297.w_25_4 ctx), ((parallel_generatedRoot_297.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_297.Facts_0  (ctx : parallel_generatedRoot_297Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_297_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_297.w_0_0 ctx), ((parallel_generatedRoot_297.w_1_0 ctx), ((parallel_generatedRoot_297.w_2_0 ctx), ((parallel_generatedRoot_297.w_3_0 ctx), ((parallel_generatedRoot_297.w_4_0 ctx), ((parallel_generatedRoot_297.w_5_0 ctx), ())))))) (((parallel_generatedRoot_297.w_6_0 ctx) i), (((parallel_generatedRoot_297.w_6_1 ctx) i), (((parallel_generatedRoot_297.w_6_2 ctx) i), (((parallel_generatedRoot_297.w_6_3 ctx) i), (((parallel_generatedRoot_297.w_6_4 ctx) i), (((parallel_generatedRoot_297.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_0 ctx) (0) (parallel_generatedRoot_297.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_0 ctx) (1) (parallel_generatedRoot_297.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_1 ctx) (0) (parallel_generatedRoot_297.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_1 ctx) (1) (parallel_generatedRoot_297.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_2 ctx) (0) (parallel_generatedRoot_297.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_2 ctx) (1) (parallel_generatedRoot_297.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_3 ctx) (0) (parallel_generatedRoot_297.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_3 ctx) (1) (parallel_generatedRoot_297.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_4 ctx) (0) (parallel_generatedRoot_297.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_4 ctx) (1) (parallel_generatedRoot_297.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_5 ctx) (0) (parallel_generatedRoot_297.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_297.w_6_5 ctx) (1) (parallel_generatedRoot_297.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_297_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_297.w_6_0 ctx), ((parallel_generatedRoot_297.w_6_1 ctx), ((parallel_generatedRoot_297.w_6_2 ctx), ((parallel_generatedRoot_297.w_6_3 ctx), ((parallel_generatedRoot_297.w_6_4 ctx), ((parallel_generatedRoot_297.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_297.w_9_0 ctx), ((parallel_generatedRoot_297.w_12_0 ctx), ((parallel_generatedRoot_297.w_15_0 ctx), ((parallel_generatedRoot_297.w_18_0 ctx), ((parallel_generatedRoot_297.w_21_0 ctx), ((parallel_generatedRoot_297.w_24_0 ctx), ())))))) ((parallel_generatedRoot_297.w_25_0 ctx), ((parallel_generatedRoot_297.w_25_1 ctx), ((parallel_generatedRoot_297.w_25_2 ctx), ((parallel_generatedRoot_297.w_25_3 ctx), ((parallel_generatedRoot_297.w_25_4 ctx), ((parallel_generatedRoot_297.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_297.outs ctx

theorem parallel_generatedRoot_297.facts_0  {ctx : parallel_generatedRoot_297Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_297.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_297.w_0_0 ctx) (parallel_generatedRoot_297.w_1_0 ctx) (parallel_generatedRoot_297.w_2_0 ctx) (parallel_generatedRoot_297.w_3_0 ctx) (parallel_generatedRoot_297.w_4_0 ctx) (parallel_generatedRoot_297.w_5_0 ctx) (parallel_generatedRoot_297.w_6_0 ctx) (parallel_generatedRoot_297.w_6_1 ctx) (parallel_generatedRoot_297.w_6_2 ctx) (parallel_generatedRoot_297.w_6_3 ctx) (parallel_generatedRoot_297.w_6_4 ctx) (parallel_generatedRoot_297.w_6_5 ctx) (parallel_generatedRoot_297.w_7_0 ctx) (parallel_generatedRoot_297.w_8_0 ctx) (parallel_generatedRoot_297.w_9_0 ctx) (parallel_generatedRoot_297.w_10_0 ctx) (parallel_generatedRoot_297.w_11_0 ctx) (parallel_generatedRoot_297.w_12_0 ctx) (parallel_generatedRoot_297.w_13_0 ctx) (parallel_generatedRoot_297.w_14_0 ctx) (parallel_generatedRoot_297.w_15_0 ctx) (parallel_generatedRoot_297.w_16_0 ctx) (parallel_generatedRoot_297.w_17_0 ctx) (parallel_generatedRoot_297.w_18_0 ctx) (parallel_generatedRoot_297.w_19_0 ctx) (parallel_generatedRoot_297.w_20_0 ctx) (parallel_generatedRoot_297.w_21_0 ctx) (parallel_generatedRoot_297.w_22_0 ctx) (parallel_generatedRoot_297.w_23_0 ctx) (parallel_generatedRoot_297.w_24_0 ctx) (parallel_generatedRoot_297.w_25_0 ctx) (parallel_generatedRoot_297.w_25_1 ctx) (parallel_generatedRoot_297.w_25_2 ctx) (parallel_generatedRoot_297.w_25_3 ctx) (parallel_generatedRoot_297.w_25_4 ctx) (parallel_generatedRoot_297.w_25_5 ctx) outputs) :
    parallel_generatedRoot_297.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_297.Facts  (ctx : parallel_generatedRoot_297Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_297.Facts_0 ctx outputs

theorem parallel_generatedRoot_297.facts  (ctx : parallel_generatedRoot_297Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_297.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_297.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_297.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_301.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_301.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_301Ctx" => (parallel_generatedRoot_301.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_301.w_0_0  (ctx : parallel_generatedRoot_301Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_301.w_1_0  (ctx : parallel_generatedRoot_301Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_301.w_2_0  (ctx : parallel_generatedRoot_301Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_301.w_3_0  (ctx : parallel_generatedRoot_301Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_301.w_4_0  (ctx : parallel_generatedRoot_301Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_301.w_5_0  (ctx : parallel_generatedRoot_301Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_301.w_6_0  (ctx : parallel_generatedRoot_301Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_301.w_6_1  (ctx : parallel_generatedRoot_301Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_301.w_6_2  (ctx : parallel_generatedRoot_301Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_301.w_6_3  (ctx : parallel_generatedRoot_301Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_301.w_6_4  (ctx : parallel_generatedRoot_301Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_301.w_6_5  (ctx : parallel_generatedRoot_301Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_301.w_7_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_301.w_8_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_301.w_9_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_301.w_7_0 ctx) (parallel_generatedRoot_301.w_8_0 ctx)

def parallel_generatedRoot_301.w_10_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_301.w_11_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_301.w_12_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_301.w_10_0 ctx) (parallel_generatedRoot_301.w_11_0 ctx)

def parallel_generatedRoot_301.w_13_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_301.w_14_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_301.w_15_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_301.w_13_0 ctx) (parallel_generatedRoot_301.w_14_0 ctx)

def parallel_generatedRoot_301.w_16_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_301.w_17_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_301.w_18_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_301.w_16_0 ctx) (parallel_generatedRoot_301.w_17_0 ctx)

def parallel_generatedRoot_301.w_19_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_301.w_20_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_301.w_21_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_301.w_19_0 ctx) (parallel_generatedRoot_301.w_20_0 ctx)

def parallel_generatedRoot_301.w_22_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_301.w_23_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_301.w_24_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_301.w_22_0 ctx) (parallel_generatedRoot_301.w_23_0 ctx)

def parallel_generatedRoot_301.w_25_0  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_301.w_25_1  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_301.w_25_2  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_301.w_25_3  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_301.w_25_4  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_301.w_25_5  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_301.outs  (ctx : parallel_generatedRoot_301Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_301.w_25_0 ctx), ((parallel_generatedRoot_301.w_25_1 ctx), ((parallel_generatedRoot_301.w_25_2 ctx), ((parallel_generatedRoot_301.w_25_3 ctx), ((parallel_generatedRoot_301.w_25_4 ctx), ((parallel_generatedRoot_301.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_301.Facts_0  (ctx : parallel_generatedRoot_301Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_301_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_301.w_0_0 ctx), ((parallel_generatedRoot_301.w_1_0 ctx), ((parallel_generatedRoot_301.w_2_0 ctx), ((parallel_generatedRoot_301.w_3_0 ctx), ((parallel_generatedRoot_301.w_4_0 ctx), ((parallel_generatedRoot_301.w_5_0 ctx), ())))))) (((parallel_generatedRoot_301.w_6_0 ctx) i), (((parallel_generatedRoot_301.w_6_1 ctx) i), (((parallel_generatedRoot_301.w_6_2 ctx) i), (((parallel_generatedRoot_301.w_6_3 ctx) i), (((parallel_generatedRoot_301.w_6_4 ctx) i), (((parallel_generatedRoot_301.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_0 ctx) (0) (parallel_generatedRoot_301.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_0 ctx) (1) (parallel_generatedRoot_301.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_1 ctx) (0) (parallel_generatedRoot_301.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_1 ctx) (1) (parallel_generatedRoot_301.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_2 ctx) (0) (parallel_generatedRoot_301.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_2 ctx) (1) (parallel_generatedRoot_301.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_3 ctx) (0) (parallel_generatedRoot_301.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_3 ctx) (1) (parallel_generatedRoot_301.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_4 ctx) (0) (parallel_generatedRoot_301.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_4 ctx) (1) (parallel_generatedRoot_301.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_5 ctx) (0) (parallel_generatedRoot_301.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_301.w_6_5 ctx) (1) (parallel_generatedRoot_301.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_301_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_301.w_6_0 ctx), ((parallel_generatedRoot_301.w_6_1 ctx), ((parallel_generatedRoot_301.w_6_2 ctx), ((parallel_generatedRoot_301.w_6_3 ctx), ((parallel_generatedRoot_301.w_6_4 ctx), ((parallel_generatedRoot_301.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_301.w_9_0 ctx), ((parallel_generatedRoot_301.w_12_0 ctx), ((parallel_generatedRoot_301.w_15_0 ctx), ((parallel_generatedRoot_301.w_18_0 ctx), ((parallel_generatedRoot_301.w_21_0 ctx), ((parallel_generatedRoot_301.w_24_0 ctx), ())))))) ((parallel_generatedRoot_301.w_25_0 ctx), ((parallel_generatedRoot_301.w_25_1 ctx), ((parallel_generatedRoot_301.w_25_2 ctx), ((parallel_generatedRoot_301.w_25_3 ctx), ((parallel_generatedRoot_301.w_25_4 ctx), ((parallel_generatedRoot_301.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_301.outs ctx

theorem parallel_generatedRoot_301.facts_0  {ctx : parallel_generatedRoot_301Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_301.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_301.w_0_0 ctx) (parallel_generatedRoot_301.w_1_0 ctx) (parallel_generatedRoot_301.w_2_0 ctx) (parallel_generatedRoot_301.w_3_0 ctx) (parallel_generatedRoot_301.w_4_0 ctx) (parallel_generatedRoot_301.w_5_0 ctx) (parallel_generatedRoot_301.w_6_0 ctx) (parallel_generatedRoot_301.w_6_1 ctx) (parallel_generatedRoot_301.w_6_2 ctx) (parallel_generatedRoot_301.w_6_3 ctx) (parallel_generatedRoot_301.w_6_4 ctx) (parallel_generatedRoot_301.w_6_5 ctx) (parallel_generatedRoot_301.w_7_0 ctx) (parallel_generatedRoot_301.w_8_0 ctx) (parallel_generatedRoot_301.w_9_0 ctx) (parallel_generatedRoot_301.w_10_0 ctx) (parallel_generatedRoot_301.w_11_0 ctx) (parallel_generatedRoot_301.w_12_0 ctx) (parallel_generatedRoot_301.w_13_0 ctx) (parallel_generatedRoot_301.w_14_0 ctx) (parallel_generatedRoot_301.w_15_0 ctx) (parallel_generatedRoot_301.w_16_0 ctx) (parallel_generatedRoot_301.w_17_0 ctx) (parallel_generatedRoot_301.w_18_0 ctx) (parallel_generatedRoot_301.w_19_0 ctx) (parallel_generatedRoot_301.w_20_0 ctx) (parallel_generatedRoot_301.w_21_0 ctx) (parallel_generatedRoot_301.w_22_0 ctx) (parallel_generatedRoot_301.w_23_0 ctx) (parallel_generatedRoot_301.w_24_0 ctx) (parallel_generatedRoot_301.w_25_0 ctx) (parallel_generatedRoot_301.w_25_1 ctx) (parallel_generatedRoot_301.w_25_2 ctx) (parallel_generatedRoot_301.w_25_3 ctx) (parallel_generatedRoot_301.w_25_4 ctx) (parallel_generatedRoot_301.w_25_5 ctx) outputs) :
    parallel_generatedRoot_301.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_301.Facts  (ctx : parallel_generatedRoot_301Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_301.Facts_0 ctx outputs

theorem parallel_generatedRoot_301.facts  (ctx : parallel_generatedRoot_301Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_301.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_301.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_301.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_305.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_305.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_305Ctx" => (parallel_generatedRoot_305.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_305.w_0_0  (ctx : parallel_generatedRoot_305Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_305.w_1_0  (ctx : parallel_generatedRoot_305Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_305.w_2_0  (ctx : parallel_generatedRoot_305Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_305.w_3_0  (ctx : parallel_generatedRoot_305Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_305.w_4_0  (ctx : parallel_generatedRoot_305Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_305.w_5_0  (ctx : parallel_generatedRoot_305Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_305.w_6_0  (ctx : parallel_generatedRoot_305Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_305.w_6_1  (ctx : parallel_generatedRoot_305Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_305.w_6_2  (ctx : parallel_generatedRoot_305Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_305.w_6_3  (ctx : parallel_generatedRoot_305Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_305.w_6_4  (ctx : parallel_generatedRoot_305Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_305.w_6_5  (ctx : parallel_generatedRoot_305Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_305.w_7_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_305.w_8_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_305.w_9_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_305.w_7_0 ctx) (parallel_generatedRoot_305.w_8_0 ctx)

def parallel_generatedRoot_305.w_10_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_305.w_11_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_305.w_12_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_305.w_10_0 ctx) (parallel_generatedRoot_305.w_11_0 ctx)

def parallel_generatedRoot_305.w_13_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_305.w_14_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_305.w_15_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_305.w_13_0 ctx) (parallel_generatedRoot_305.w_14_0 ctx)

def parallel_generatedRoot_305.w_16_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_305.w_17_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_305.w_18_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_305.w_16_0 ctx) (parallel_generatedRoot_305.w_17_0 ctx)

def parallel_generatedRoot_305.w_19_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_305.w_20_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_305.w_21_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_305.w_19_0 ctx) (parallel_generatedRoot_305.w_20_0 ctx)

def parallel_generatedRoot_305.w_22_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_305.w_23_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_305.w_24_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_305.w_22_0 ctx) (parallel_generatedRoot_305.w_23_0 ctx)

def parallel_generatedRoot_305.w_25_0  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_305.w_25_1  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_305.w_25_2  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_305.w_25_3  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_305.w_25_4  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_305.w_25_5  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_305.outs  (ctx : parallel_generatedRoot_305Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_305.w_25_0 ctx), ((parallel_generatedRoot_305.w_25_1 ctx), ((parallel_generatedRoot_305.w_25_2 ctx), ((parallel_generatedRoot_305.w_25_3 ctx), ((parallel_generatedRoot_305.w_25_4 ctx), ((parallel_generatedRoot_305.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_305.Facts_0  (ctx : parallel_generatedRoot_305Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_305_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_305.w_0_0 ctx), ((parallel_generatedRoot_305.w_1_0 ctx), ((parallel_generatedRoot_305.w_2_0 ctx), ((parallel_generatedRoot_305.w_3_0 ctx), ((parallel_generatedRoot_305.w_4_0 ctx), ((parallel_generatedRoot_305.w_5_0 ctx), ())))))) (((parallel_generatedRoot_305.w_6_0 ctx) i), (((parallel_generatedRoot_305.w_6_1 ctx) i), (((parallel_generatedRoot_305.w_6_2 ctx) i), (((parallel_generatedRoot_305.w_6_3 ctx) i), (((parallel_generatedRoot_305.w_6_4 ctx) i), (((parallel_generatedRoot_305.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_0 ctx) (0) (parallel_generatedRoot_305.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_0 ctx) (1) (parallel_generatedRoot_305.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_1 ctx) (0) (parallel_generatedRoot_305.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_1 ctx) (1) (parallel_generatedRoot_305.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_2 ctx) (0) (parallel_generatedRoot_305.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_2 ctx) (1) (parallel_generatedRoot_305.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_3 ctx) (0) (parallel_generatedRoot_305.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_3 ctx) (1) (parallel_generatedRoot_305.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_4 ctx) (0) (parallel_generatedRoot_305.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_4 ctx) (1) (parallel_generatedRoot_305.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_5 ctx) (0) (parallel_generatedRoot_305.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_305.w_6_5 ctx) (1) (parallel_generatedRoot_305.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_305_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_305.w_6_0 ctx), ((parallel_generatedRoot_305.w_6_1 ctx), ((parallel_generatedRoot_305.w_6_2 ctx), ((parallel_generatedRoot_305.w_6_3 ctx), ((parallel_generatedRoot_305.w_6_4 ctx), ((parallel_generatedRoot_305.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_305.w_9_0 ctx), ((parallel_generatedRoot_305.w_12_0 ctx), ((parallel_generatedRoot_305.w_15_0 ctx), ((parallel_generatedRoot_305.w_18_0 ctx), ((parallel_generatedRoot_305.w_21_0 ctx), ((parallel_generatedRoot_305.w_24_0 ctx), ())))))) ((parallel_generatedRoot_305.w_25_0 ctx), ((parallel_generatedRoot_305.w_25_1 ctx), ((parallel_generatedRoot_305.w_25_2 ctx), ((parallel_generatedRoot_305.w_25_3 ctx), ((parallel_generatedRoot_305.w_25_4 ctx), ((parallel_generatedRoot_305.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_305.outs ctx

theorem parallel_generatedRoot_305.facts_0  {ctx : parallel_generatedRoot_305Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_305.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_305.w_0_0 ctx) (parallel_generatedRoot_305.w_1_0 ctx) (parallel_generatedRoot_305.w_2_0 ctx) (parallel_generatedRoot_305.w_3_0 ctx) (parallel_generatedRoot_305.w_4_0 ctx) (parallel_generatedRoot_305.w_5_0 ctx) (parallel_generatedRoot_305.w_6_0 ctx) (parallel_generatedRoot_305.w_6_1 ctx) (parallel_generatedRoot_305.w_6_2 ctx) (parallel_generatedRoot_305.w_6_3 ctx) (parallel_generatedRoot_305.w_6_4 ctx) (parallel_generatedRoot_305.w_6_5 ctx) (parallel_generatedRoot_305.w_7_0 ctx) (parallel_generatedRoot_305.w_8_0 ctx) (parallel_generatedRoot_305.w_9_0 ctx) (parallel_generatedRoot_305.w_10_0 ctx) (parallel_generatedRoot_305.w_11_0 ctx) (parallel_generatedRoot_305.w_12_0 ctx) (parallel_generatedRoot_305.w_13_0 ctx) (parallel_generatedRoot_305.w_14_0 ctx) (parallel_generatedRoot_305.w_15_0 ctx) (parallel_generatedRoot_305.w_16_0 ctx) (parallel_generatedRoot_305.w_17_0 ctx) (parallel_generatedRoot_305.w_18_0 ctx) (parallel_generatedRoot_305.w_19_0 ctx) (parallel_generatedRoot_305.w_20_0 ctx) (parallel_generatedRoot_305.w_21_0 ctx) (parallel_generatedRoot_305.w_22_0 ctx) (parallel_generatedRoot_305.w_23_0 ctx) (parallel_generatedRoot_305.w_24_0 ctx) (parallel_generatedRoot_305.w_25_0 ctx) (parallel_generatedRoot_305.w_25_1 ctx) (parallel_generatedRoot_305.w_25_2 ctx) (parallel_generatedRoot_305.w_25_3 ctx) (parallel_generatedRoot_305.w_25_4 ctx) (parallel_generatedRoot_305.w_25_5 ctx) outputs) :
    parallel_generatedRoot_305.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_305.Facts  (ctx : parallel_generatedRoot_305Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_305.Facts_0 ctx outputs

theorem parallel_generatedRoot_305.facts  (ctx : parallel_generatedRoot_305Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_305.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_305.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_305.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_309.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × (Fin 2607022080 → mxxMat0) × Unit
  witness : parallel_generatedRoot_309.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_309Ctx" => (parallel_generatedRoot_309.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_309.w_0_0  (ctx : parallel_generatedRoot_309Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_309.w_1_0  (ctx : parallel_generatedRoot_309Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_309.w_2_0  (ctx : parallel_generatedRoot_309Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_309.w_3_0  (ctx : parallel_generatedRoot_309Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_309.w_4_0  (ctx : parallel_generatedRoot_309Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_309.w_5_0  (ctx : parallel_generatedRoot_309Ctx) : Fin 2607022080 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_309.w_6_0  (ctx : parallel_generatedRoot_309Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_309.w_6_1  (ctx : parallel_generatedRoot_309Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_309.w_6_2  (ctx : parallel_generatedRoot_309Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_309.w_6_3  (ctx : parallel_generatedRoot_309Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_309.w_6_4  (ctx : parallel_generatedRoot_309Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_309.w_6_5  (ctx : parallel_generatedRoot_309Ctx) : Fin 130 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_309.w_7_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_309.w_8_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_309.w_9_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_309.w_7_0 ctx) (parallel_generatedRoot_309.w_8_0 ctx)

def parallel_generatedRoot_309.w_10_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_309.w_11_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_309.w_12_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_309.w_10_0 ctx) (parallel_generatedRoot_309.w_11_0 ctx)

def parallel_generatedRoot_309.w_13_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_309.w_14_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_309.w_15_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_309.w_13_0 ctx) (parallel_generatedRoot_309.w_14_0 ctx)

def parallel_generatedRoot_309.w_16_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_309.w_17_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_309.w_18_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_309.w_16_0 ctx) (parallel_generatedRoot_309.w_17_0 ctx)

def parallel_generatedRoot_309.w_19_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_309.w_20_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_309.w_21_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_309.w_19_0 ctx) (parallel_generatedRoot_309.w_20_0 ctx)

def parallel_generatedRoot_309.w_22_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_309.w_23_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_309.w_24_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_309.w_22_0 ctx) (parallel_generatedRoot_309.w_23_0 ctx)

def parallel_generatedRoot_309.w_25_0  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_309.w_25_1  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_309.w_25_2  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_309.w_25_3  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_309.w_25_4  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_309.w_25_5  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_309.outs  (ctx : parallel_generatedRoot_309Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_309.w_25_0 ctx), ((parallel_generatedRoot_309.w_25_1 ctx), ((parallel_generatedRoot_309.w_25_2 ctx), ((parallel_generatedRoot_309.w_25_3 ctx), ((parallel_generatedRoot_309.w_25_4 ctx), ((parallel_generatedRoot_309.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_309.Facts_0  (ctx : parallel_generatedRoot_309Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (130) = 130
  h6_5 : (∀ i : Fin 130, parallel_parallel_generatedRoot_309_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_309.w_0_0 ctx), ((parallel_generatedRoot_309.w_1_0 ctx), ((parallel_generatedRoot_309.w_2_0 ctx), ((parallel_generatedRoot_309.w_3_0 ctx), ((parallel_generatedRoot_309.w_4_0 ctx), ((parallel_generatedRoot_309.w_5_0 ctx), ())))))) (((parallel_generatedRoot_309.w_6_0 ctx) i), (((parallel_generatedRoot_309.w_6_1 ctx) i), (((parallel_generatedRoot_309.w_6_2 ctx) i), (((parallel_generatedRoot_309.w_6_3 ctx) i), (((parallel_generatedRoot_309.w_6_4 ctx) i), (((parallel_generatedRoot_309.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_0 ctx) (0) (parallel_generatedRoot_309.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_0 ctx) (1) (parallel_generatedRoot_309.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_1 ctx) (0) (parallel_generatedRoot_309.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_1 ctx) (1) (parallel_generatedRoot_309.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_2 ctx) (0) (parallel_generatedRoot_309.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_2 ctx) (1) (parallel_generatedRoot_309.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_3 ctx) (0) (parallel_generatedRoot_309.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_3 ctx) (1) (parallel_generatedRoot_309.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_4 ctx) (0) (parallel_generatedRoot_309.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_4 ctx) (1) (parallel_generatedRoot_309.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_5 ctx) (0) (parallel_generatedRoot_309.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_309.w_6_5 ctx) (1) (parallel_generatedRoot_309.w_23_0 ctx)
  t14 : 0 ≤ (128)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_309_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_309.w_6_0 ctx), ((parallel_generatedRoot_309.w_6_1 ctx), ((parallel_generatedRoot_309.w_6_2 ctx), ((parallel_generatedRoot_309.w_6_3 ctx), ((parallel_generatedRoot_309.w_6_4 ctx), ((parallel_generatedRoot_309.w_6_5 ctx), ())))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_309.w_9_0 ctx), ((parallel_generatedRoot_309.w_12_0 ctx), ((parallel_generatedRoot_309.w_15_0 ctx), ((parallel_generatedRoot_309.w_18_0 ctx), ((parallel_generatedRoot_309.w_21_0 ctx), ((parallel_generatedRoot_309.w_24_0 ctx), ())))))) ((parallel_generatedRoot_309.w_25_0 ctx), ((parallel_generatedRoot_309.w_25_1 ctx), ((parallel_generatedRoot_309.w_25_2 ctx), ((parallel_generatedRoot_309.w_25_3 ctx), ((parallel_generatedRoot_309.w_25_4 ctx), ((parallel_generatedRoot_309.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_309.outs ctx

theorem parallel_generatedRoot_309.facts_0  {ctx : parallel_generatedRoot_309Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_309.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_309.w_0_0 ctx) (parallel_generatedRoot_309.w_1_0 ctx) (parallel_generatedRoot_309.w_2_0 ctx) (parallel_generatedRoot_309.w_3_0 ctx) (parallel_generatedRoot_309.w_4_0 ctx) (parallel_generatedRoot_309.w_5_0 ctx) (parallel_generatedRoot_309.w_6_0 ctx) (parallel_generatedRoot_309.w_6_1 ctx) (parallel_generatedRoot_309.w_6_2 ctx) (parallel_generatedRoot_309.w_6_3 ctx) (parallel_generatedRoot_309.w_6_4 ctx) (parallel_generatedRoot_309.w_6_5 ctx) (parallel_generatedRoot_309.w_7_0 ctx) (parallel_generatedRoot_309.w_8_0 ctx) (parallel_generatedRoot_309.w_9_0 ctx) (parallel_generatedRoot_309.w_10_0 ctx) (parallel_generatedRoot_309.w_11_0 ctx) (parallel_generatedRoot_309.w_12_0 ctx) (parallel_generatedRoot_309.w_13_0 ctx) (parallel_generatedRoot_309.w_14_0 ctx) (parallel_generatedRoot_309.w_15_0 ctx) (parallel_generatedRoot_309.w_16_0 ctx) (parallel_generatedRoot_309.w_17_0 ctx) (parallel_generatedRoot_309.w_18_0 ctx) (parallel_generatedRoot_309.w_19_0 ctx) (parallel_generatedRoot_309.w_20_0 ctx) (parallel_generatedRoot_309.w_21_0 ctx) (parallel_generatedRoot_309.w_22_0 ctx) (parallel_generatedRoot_309.w_23_0 ctx) (parallel_generatedRoot_309.w_24_0 ctx) (parallel_generatedRoot_309.w_25_0 ctx) (parallel_generatedRoot_309.w_25_1 ctx) (parallel_generatedRoot_309.w_25_2 ctx) (parallel_generatedRoot_309.w_25_3 ctx) (parallel_generatedRoot_309.w_25_4 ctx) (parallel_generatedRoot_309.w_25_5 ctx) outputs) :
    parallel_generatedRoot_309.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_309.Facts  (ctx : parallel_generatedRoot_309Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_309.Facts_0 ctx outputs

theorem parallel_generatedRoot_309.facts  (ctx : parallel_generatedRoot_309Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_309.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_309.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_309.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_310.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_310.InputType)
  witness : parallel_generatedRoot_310.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_310Ctx" => (parallel_generatedRoot_310.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_310.w_1_0  (ctx : parallel_generatedRoot_310Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_310.w_3_0  (ctx : parallel_generatedRoot_310Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_310.w_5_0  (ctx : parallel_generatedRoot_310Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_310.w_7_0  (ctx : parallel_generatedRoot_310Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_310.w_9_0  (ctx : parallel_generatedRoot_310Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_310.w_10_0  (ctx : parallel_generatedRoot_310Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_310.w_11_0  (ctx : parallel_generatedRoot_310Ctx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_310.w_12_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_310.w_13_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_310.w_14_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_310.w_15_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_310.w_16_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_310.w_17_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_310.w_0_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_310.w_2_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_310.w_4_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_310.w_6_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_310.w_8_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_310.w_18_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_310.w_18_1  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_310.w_18_2  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_310.w_18_3  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_310.w_18_4  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_310.w_18_5  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_310.w_18_6  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_310.w_18_7  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_310.w_18_8  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_310.w_18_9  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_310.w_19_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_310.w_20_0  (ctx : parallel_generatedRoot_310Ctx) : Int :=
  (MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])

def parallel_generatedRoot_310.w_21_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_310.w_22_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_310.w_23_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_310.w_23_concat_1  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_310.w_24_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_310.w_25_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_310.w_26_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_310.w_27_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_310.w_27_concat_1  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_310.w_28_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_310.w_29_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_310.w_30_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_310.w_31_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_310.w_32_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_310.w_32_concat_1  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_310.w_33_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_310.w_34_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_310.w_35_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_310.w_36_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_310.w_36_concat_1  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_310.w_37_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_310.w_38_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_310.w_39_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_310.w_40_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_310.w_41_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_310.w_37_0 ctx) (parallel_generatedRoot_310.w_40_0 ctx)

def parallel_generatedRoot_310.w_42_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_310.w_43_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_310.w_44_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_310.w_45_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_310.w_46_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_310.w_47_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_310.w_28_0 ctx) (parallel_generatedRoot_310.w_46_0 ctx)

def parallel_generatedRoot_310.w_48_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_310.w_49_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_18_2 ctx) (parallel_generatedRoot_310.w_48_0 ctx)

def parallel_generatedRoot_310.w_50_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_310.w_51_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_310.w_52_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_310.w_53_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_310.w_37_0 ctx) (parallel_generatedRoot_310.w_52_0 ctx)

def parallel_generatedRoot_310.w_54_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_310.w_55_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_310.w_56_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_310.w_57_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_310.w_58_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_310.w_59_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_310.w_28_0 ctx) (parallel_generatedRoot_310.w_58_0 ctx)

def parallel_generatedRoot_310.w_60_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_310.w_61_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_18_3 ctx) (parallel_generatedRoot_310.w_60_0 ctx)

def parallel_generatedRoot_310.w_62_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_310.w_63_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_18_4 ctx) (parallel_generatedRoot_310.w_62_0 ctx)

def parallel_generatedRoot_310.w_64_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_310.w_65_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_18_5 ctx) (parallel_generatedRoot_310.w_64_0 ctx)

def parallel_generatedRoot_310.w_66_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_310.w_67_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_310.w_68_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_66_0 ctx) (parallel_generatedRoot_310.w_67_0 ctx)

def parallel_generatedRoot_310.w_69_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_18_0 ctx) (parallel_generatedRoot_310.w_68_0 ctx)

def parallel_generatedRoot_310.w_70_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_310.w_71_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_310.w_72_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_70_0 ctx) (parallel_generatedRoot_310.w_71_0 ctx)

def parallel_generatedRoot_310.w_73_0  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_310.w_18_1 ctx) (parallel_generatedRoot_310.w_72_0 ctx)

def parallel_generatedRoot_310.outs  (ctx : parallel_generatedRoot_310Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_310.w_49_0 ctx), ((parallel_generatedRoot_310.w_61_0 ctx), ((parallel_generatedRoot_310.w_63_0 ctx), ((parallel_generatedRoot_310.w_65_0 ctx), ((parallel_generatedRoot_310.w_69_0 ctx), ((parallel_generatedRoot_310.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_310.Facts_0  (ctx : parallel_generatedRoot_310Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_310.w_1_0 ctx) (0) (parallel_generatedRoot_310.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_310.w_3_0 ctx) (0) (parallel_generatedRoot_310.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_310.w_5_0 ctx) (0) (parallel_generatedRoot_310.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_310.w_7_0 ctx) (0) (parallel_generatedRoot_310.w_8_0 ctx)
  t4 : 0 ≤ (127)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_310_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_310.w_9_0 ctx), ((parallel_generatedRoot_310.w_10_0 ctx), ((parallel_generatedRoot_310.w_11_0 ctx), ((parallel_generatedRoot_310.w_12_0 ctx), ((parallel_generatedRoot_310.w_13_0 ctx), ((parallel_generatedRoot_310.w_14_0 ctx), ((parallel_generatedRoot_310.w_15_0 ctx), ((parallel_generatedRoot_310.w_16_0 ctx), ((parallel_generatedRoot_310.w_17_0 ctx), ((parallel_generatedRoot_310.w_1_0 ctx), ((parallel_generatedRoot_310.w_3_0 ctx), ((parallel_generatedRoot_310.w_5_0 ctx), ((parallel_generatedRoot_310.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (127)) ((parallel_generatedRoot_310.w_0_0 ctx), ((parallel_generatedRoot_310.w_0_0 ctx), ((parallel_generatedRoot_310.w_0_0 ctx), ((parallel_generatedRoot_310.w_0_0 ctx), ((parallel_generatedRoot_310.w_0_0 ctx), ((parallel_generatedRoot_310.w_0_0 ctx), ((parallel_generatedRoot_310.w_2_0 ctx), ((parallel_generatedRoot_310.w_4_0 ctx), ((parallel_generatedRoot_310.w_6_0 ctx), ((parallel_generatedRoot_310.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_310.w_18_0 ctx), ((parallel_generatedRoot_310.w_18_1 ctx), ((parallel_generatedRoot_310.w_18_2 ctx), ((parallel_generatedRoot_310.w_18_3 ctx), ((parallel_generatedRoot_310.w_18_4 ctx), ((parallel_generatedRoot_310.w_18_5 ctx), ((parallel_generatedRoot_310.w_18_6 ctx), ((parallel_generatedRoot_310.w_18_7 ctx), ((parallel_generatedRoot_310.w_18_8 ctx), ((parallel_generatedRoot_310.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_310.w_9_0 ctx) (127) (parallel_generatedRoot_310.w_19_0 ctx)
  t7 : 0 ≤ (127)
  t8 : (127) < 128
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_310.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_310.w_10_0 ctx)) (parallel_generatedRoot_310.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_310.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_310.w_10_0 ctx)) (parallel_generatedRoot_310.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_19_0 ctx) (parallel_generatedRoot_310.w_21_0 ctx) (parallel_generatedRoot_310.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_23_concat_1 ctx) (parallel_generatedRoot_310.w_22_0 ctx) (parallel_generatedRoot_310.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_310.w_11_0 ctx) (127) (parallel_generatedRoot_310.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_310.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_310.w_10_0 ctx)) (parallel_generatedRoot_310.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_310.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_310.w_10_0 ctx)) (parallel_generatedRoot_310.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_24_0 ctx) (parallel_generatedRoot_310.w_25_0 ctx) (parallel_generatedRoot_310.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_27_concat_1 ctx) (parallel_generatedRoot_310.w_26_0 ctx) (parallel_generatedRoot_310.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_310.w_23_0 ctx) (parallel_generatedRoot_310.w_27_0 ctx) (parallel_generatedRoot_310.w_28_0 ctx)
  t19 : 0 ≤ (127)
  t20 : (127) < 128
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_310.w_12_0 ctx) (parallel_generatedRoot_310.w_29_0 ctx)
  t22 : 0 ≤ (127)
  t23 : (127) < 128
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_310.w_13_0 ctx) (parallel_generatedRoot_310.w_30_0 ctx)
  t25 : 0 ≤ (127)
  t26 : (127) < 128
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_310.w_14_0 ctx) (parallel_generatedRoot_310.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_29_0 ctx) (parallel_generatedRoot_310.w_30_0 ctx) (parallel_generatedRoot_310.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_32_concat_1 ctx) (parallel_generatedRoot_310.w_31_0 ctx) (parallel_generatedRoot_310.w_32_0 ctx)
  t30 : 0 ≤ (127)
  t31 : (127) < 128

theorem parallel_generatedRoot_310.facts_0  {ctx : parallel_generatedRoot_310Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_310.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_310.w_1_0 ctx) (parallel_generatedRoot_310.w_3_0 ctx) (parallel_generatedRoot_310.w_5_0 ctx) (parallel_generatedRoot_310.w_7_0 ctx) (parallel_generatedRoot_310.w_9_0 ctx) (parallel_generatedRoot_310.w_10_0 ctx) (parallel_generatedRoot_310.w_11_0 ctx) (parallel_generatedRoot_310.w_12_0 ctx) (parallel_generatedRoot_310.w_13_0 ctx) (parallel_generatedRoot_310.w_14_0 ctx) (parallel_generatedRoot_310.w_15_0 ctx) (parallel_generatedRoot_310.w_16_0 ctx) (parallel_generatedRoot_310.w_17_0 ctx) (parallel_generatedRoot_310.w_0_0 ctx) (parallel_generatedRoot_310.w_2_0 ctx) (parallel_generatedRoot_310.w_4_0 ctx) (parallel_generatedRoot_310.w_6_0 ctx) (parallel_generatedRoot_310.w_8_0 ctx) (parallel_generatedRoot_310.w_18_0 ctx) (parallel_generatedRoot_310.w_18_1 ctx) (parallel_generatedRoot_310.w_18_2 ctx) (parallel_generatedRoot_310.w_18_3 ctx) (parallel_generatedRoot_310.w_18_4 ctx) (parallel_generatedRoot_310.w_18_5 ctx) (parallel_generatedRoot_310.w_18_6 ctx) (parallel_generatedRoot_310.w_18_7 ctx) (parallel_generatedRoot_310.w_18_8 ctx) (parallel_generatedRoot_310.w_18_9 ctx) (parallel_generatedRoot_310.w_19_0 ctx) (parallel_generatedRoot_310.w_20_0 ctx) (parallel_generatedRoot_310.w_21_0 ctx) (parallel_generatedRoot_310.w_22_0 ctx) (parallel_generatedRoot_310.w_23_0 ctx) (parallel_generatedRoot_310.w_23_concat_1 ctx) (parallel_generatedRoot_310.w_24_0 ctx) (parallel_generatedRoot_310.w_25_0 ctx) (parallel_generatedRoot_310.w_26_0 ctx) (parallel_generatedRoot_310.w_27_0 ctx) (parallel_generatedRoot_310.w_27_concat_1 ctx) (parallel_generatedRoot_310.w_28_0 ctx) (parallel_generatedRoot_310.w_29_0 ctx) (parallel_generatedRoot_310.w_30_0 ctx) (parallel_generatedRoot_310.w_31_0 ctx) (parallel_generatedRoot_310.w_32_0 ctx) (parallel_generatedRoot_310.w_32_concat_1 ctx)) :
    parallel_generatedRoot_310.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_310.Facts_1  (ctx : parallel_generatedRoot_310Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_310.w_15_0 ctx) (parallel_generatedRoot_310.w_33_0 ctx)
  t33 : 0 ≤ (127)
  t34 : (127) < 128
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_310.w_16_0 ctx) (parallel_generatedRoot_310.w_34_0 ctx)
  t36 : 0 ≤ (127)
  t37 : (127) < 128
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (127) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_310.w_17_0 ctx) (parallel_generatedRoot_310.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_33_0 ctx) (parallel_generatedRoot_310.w_34_0 ctx) (parallel_generatedRoot_310.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_36_concat_1 ctx) (parallel_generatedRoot_310.w_35_0 ctx) (parallel_generatedRoot_310.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_310.w_32_0 ctx) (parallel_generatedRoot_310.w_36_0 ctx) (parallel_generatedRoot_310.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_18_6 ctx) (parallel_generatedRoot_310.w_18_8 ctx) (parallel_generatedRoot_310.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_310.w_38_0 ctx) (parallel_generatedRoot_310.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_310.w_39_0 ctx) (parallel_generatedRoot_310.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 270
  t50 : 270 ≤ 270
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_41_0 ctx) 1 2 0 270 (parallel_generatedRoot_310.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 270
  t57 : 270 ≤ 270
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_41_0 ctx) 2 3 0 270 (parallel_generatedRoot_310.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_42_0 ctx) (parallel_generatedRoot_310.w_43_0 ctx) (parallel_generatedRoot_310.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_310.w_44_0 ctx) (parallel_generatedRoot_310.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_310.w_45_0 ctx) (parallel_generatedRoot_310.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_310.facts_1  {ctx : parallel_generatedRoot_310Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_310.constraints_1 backend (parallel_generatedRoot_310.w_15_0 ctx) (parallel_generatedRoot_310.w_16_0 ctx) (parallel_generatedRoot_310.w_17_0 ctx) (parallel_generatedRoot_310.w_18_6 ctx) (parallel_generatedRoot_310.w_18_8 ctx) (parallel_generatedRoot_310.w_32_0 ctx) (parallel_generatedRoot_310.w_33_0 ctx) (parallel_generatedRoot_310.w_34_0 ctx) (parallel_generatedRoot_310.w_35_0 ctx) (parallel_generatedRoot_310.w_36_0 ctx) (parallel_generatedRoot_310.w_36_concat_1 ctx) (parallel_generatedRoot_310.w_37_0 ctx) (parallel_generatedRoot_310.w_38_0 ctx) (parallel_generatedRoot_310.w_39_0 ctx) (parallel_generatedRoot_310.w_40_0 ctx) (parallel_generatedRoot_310.w_41_0 ctx) (parallel_generatedRoot_310.w_42_0 ctx) (parallel_generatedRoot_310.w_43_0 ctx) (parallel_generatedRoot_310.w_44_0 ctx) (parallel_generatedRoot_310.w_45_0 ctx) (parallel_generatedRoot_310.w_46_0 ctx)) :
    parallel_generatedRoot_310.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_310.Facts_2  (ctx : parallel_generatedRoot_310Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 270
  t67 : 270 ≤ 270
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_47_0 ctx) 1 2 0 270 (parallel_generatedRoot_310.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_18_7 ctx) (parallel_generatedRoot_310.w_18_9 ctx) (parallel_generatedRoot_310.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_310.w_50_0 ctx) (parallel_generatedRoot_310.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_310.w_51_0 ctx) (parallel_generatedRoot_310.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 270
  t77 : 270 ≤ 270
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_53_0 ctx) 1 2 0 270 (parallel_generatedRoot_310.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 270
  t84 : 270 ≤ 270
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_53_0 ctx) 2 3 0 270 (parallel_generatedRoot_310.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_310.w_54_0 ctx) (parallel_generatedRoot_310.w_55_0 ctx) (parallel_generatedRoot_310.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (parallel_generatedRoot_310.w_56_0 ctx) (parallel_generatedRoot_310.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 32 (parallel_generatedRoot_310.w_57_0 ctx) (parallel_generatedRoot_310.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 270
  t94 : 270 ≤ 270
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_59_0 ctx) 1 2 0 270 (parallel_generatedRoot_310.w_60_0 ctx)

theorem parallel_generatedRoot_310.facts_2  {ctx : parallel_generatedRoot_310Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_310.constraints_2 backend (parallel_generatedRoot_310.w_18_7 ctx) (parallel_generatedRoot_310.w_18_9 ctx) (parallel_generatedRoot_310.w_47_0 ctx) (parallel_generatedRoot_310.w_48_0 ctx) (parallel_generatedRoot_310.w_50_0 ctx) (parallel_generatedRoot_310.w_51_0 ctx) (parallel_generatedRoot_310.w_52_0 ctx) (parallel_generatedRoot_310.w_53_0 ctx) (parallel_generatedRoot_310.w_54_0 ctx) (parallel_generatedRoot_310.w_55_0 ctx) (parallel_generatedRoot_310.w_56_0 ctx) (parallel_generatedRoot_310.w_57_0 ctx) (parallel_generatedRoot_310.w_58_0 ctx) (parallel_generatedRoot_310.w_59_0 ctx) (parallel_generatedRoot_310.w_60_0 ctx)) :
    parallel_generatedRoot_310.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_310.Facts_3  (ctx : parallel_generatedRoot_310Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 270
  t101 : 270 ≤ 270
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_47_0 ctx) 2 3 0 270 (parallel_generatedRoot_310.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 270
  t108 : 270 ≤ 270
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_59_0 ctx) 2 3 0 270 (parallel_generatedRoot_310.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 270
  t115 : 270 ≤ 270
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_47_0 ctx) 0 1 0 270 (parallel_generatedRoot_310.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 270
  t122 : 270 ≤ 270
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_41_0 ctx) 0 1 0 270 (parallel_generatedRoot_310.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_310.facts_3  {ctx : parallel_generatedRoot_310Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_310.constraints_3 (parallel_generatedRoot_310.w_41_0 ctx) (parallel_generatedRoot_310.w_47_0 ctx) (parallel_generatedRoot_310.w_59_0 ctx) (parallel_generatedRoot_310.w_62_0 ctx) (parallel_generatedRoot_310.w_64_0 ctx) (parallel_generatedRoot_310.w_66_0 ctx) (parallel_generatedRoot_310.w_67_0 ctx)) :
    parallel_generatedRoot_310.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_310.Facts_4  (ctx : parallel_generatedRoot_310Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 270
  t129 : 270 ≤ 270
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_59_0 ctx) 0 1 0 270 (parallel_generatedRoot_310.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 270
  t136 : 270 ≤ 270
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_310.w_53_0 ctx) 0 1 0 270 (parallel_generatedRoot_310.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_310.outs ctx

theorem parallel_generatedRoot_310.facts_4  {ctx : parallel_generatedRoot_310Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_310.constraints_4 (parallel_generatedRoot_310.w_49_0 ctx) (parallel_generatedRoot_310.w_53_0 ctx) (parallel_generatedRoot_310.w_59_0 ctx) (parallel_generatedRoot_310.w_61_0 ctx) (parallel_generatedRoot_310.w_63_0 ctx) (parallel_generatedRoot_310.w_65_0 ctx) (parallel_generatedRoot_310.w_69_0 ctx) (parallel_generatedRoot_310.w_70_0 ctx) (parallel_generatedRoot_310.w_71_0 ctx) (parallel_generatedRoot_310.w_73_0 ctx) outputs) :
    parallel_generatedRoot_310.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_310.Facts  (ctx : parallel_generatedRoot_310Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_310.Facts_0 ctx
  c1 : parallel_generatedRoot_310.Facts_1 ctx
  c2 : parallel_generatedRoot_310.Facts_2 ctx
  c3 : parallel_generatedRoot_310.Facts_3 ctx
  c4 : parallel_generatedRoot_310.Facts_4 ctx outputs

theorem parallel_generatedRoot_310.facts  (ctx : parallel_generatedRoot_310Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_310.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_310.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_310.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_310.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_310.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_310.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_310.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_342.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (sequential_generatedRoot_342.InputType)
  witness : sequential_generatedRoot_342.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_342Ctx" => (sequential_generatedRoot_342.Ctx backend hashModel tape path params i_0)

def sequential_generatedRoot_342.w_0_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.inputs.1

def sequential_generatedRoot_342.w_31_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.inputs.2.1

def sequential_generatedRoot_342.w_35_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_342.w_39_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_342.w_43_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_342.w_47_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_342.w_2_0  (ctx : sequential_generatedRoot_342Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_342.w_8_0  (ctx : sequential_generatedRoot_342Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_342.w_12_0  (ctx : sequential_generatedRoot_342Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_342.w_16_0  (ctx : sequential_generatedRoot_342Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_342.w_20_0  (ctx : sequential_generatedRoot_342Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_342.w_24_0  (ctx : sequential_generatedRoot_342Ctx) : Fin 20054016 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_342.w_1_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_342.w_3_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  (65536 + (65536 * (Int.ofNat i_0)))

def sequential_generatedRoot_342.w_4_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_342.w_5_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  (65537 + (65536 * (Int.ofNat i_0)))

def sequential_generatedRoot_342.w_6_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_342.w_7_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_4_0 ctx) (sequential_generatedRoot_342.w_6_0 ctx)

def sequential_generatedRoot_342.w_9_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_342.w_10_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_342.w_11_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_9_0 ctx) (sequential_generatedRoot_342.w_10_0 ctx)

def sequential_generatedRoot_342.w_13_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_342.w_14_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_342.w_15_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_13_0 ctx) (sequential_generatedRoot_342.w_14_0 ctx)

def sequential_generatedRoot_342.w_17_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_342.w_18_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_342.w_19_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_17_0 ctx) (sequential_generatedRoot_342.w_18_0 ctx)

def sequential_generatedRoot_342.w_21_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_342.w_22_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def sequential_generatedRoot_342.w_23_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_21_0 ctx) (sequential_generatedRoot_342.w_22_0 ctx)

def sequential_generatedRoot_342.w_25_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_342.w_26_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def sequential_generatedRoot_342.w_27_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_25_0 ctx) (sequential_generatedRoot_342.w_26_0 ctx)

def sequential_generatedRoot_342.w_28_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_28_0

def sequential_generatedRoot_342.w_28_1  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_28_1

def sequential_generatedRoot_342.w_28_2  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_28_2

def sequential_generatedRoot_342.w_28_3  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_28_3

def sequential_generatedRoot_342.w_28_4  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_28_4

def sequential_generatedRoot_342.w_28_5  (ctx : sequential_generatedRoot_342Ctx) : mxxMat0 :=
  ctx.witness.w_28_5

def sequential_generatedRoot_342.w_29_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_342.w_30_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_0_0 ctx) (sequential_generatedRoot_342.w_29_0 ctx)

def sequential_generatedRoot_342.w_32_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_342.w_33_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_342.w_34_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_31_0 ctx) (sequential_generatedRoot_342.w_33_0 ctx)

def sequential_generatedRoot_342.w_36_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_342.w_37_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.witness.w_37_0

def sequential_generatedRoot_342.w_38_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_35_0 ctx) (sequential_generatedRoot_342.w_37_0 ctx)

def sequential_generatedRoot_342.w_40_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_342.w_41_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.witness.w_41_0

def sequential_generatedRoot_342.w_42_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_39_0 ctx) (sequential_generatedRoot_342.w_41_0 ctx)

def sequential_generatedRoot_342.w_44_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_342.w_45_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.witness.w_45_0

def sequential_generatedRoot_342.w_46_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_43_0 ctx) (sequential_generatedRoot_342.w_45_0 ctx)

def sequential_generatedRoot_342.w_48_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat6 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat6)

def sequential_generatedRoot_342.w_49_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  ctx.witness.w_49_0

def sequential_generatedRoot_342.w_50_0  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_342.w_47_0 ctx) (sequential_generatedRoot_342.w_49_0 ctx)

def sequential_generatedRoot_342.outs  (ctx : sequential_generatedRoot_342Ctx) : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit :=
  ((sequential_generatedRoot_342.w_30_0 ctx), ((sequential_generatedRoot_342.w_34_0 ctx), ((sequential_generatedRoot_342.w_38_0 ctx), ((sequential_generatedRoot_342.w_42_0 ctx), ((sequential_generatedRoot_342.w_46_0 ctx), ((sequential_generatedRoot_342.w_50_0 ctx), ()))))))

structure sequential_generatedRoot_342.Facts_0  (ctx : sequential_generatedRoot_342Ctx) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 306
  h3_0 : 0 ≤ (sequential_generatedRoot_342.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_342.w_3_0 ctx) < 20054016
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_2_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_342.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_342.w_5_0 ctx) < 20054016
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_2_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_342.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_342.w_3_0 ctx) < 20054016
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_8_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_342.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_342.w_5_0 ctx) < 20054016
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_8_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_342.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_342.w_3_0 ctx) < 20054016
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_12_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_342.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_342.w_5_0 ctx) < 20054016
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_12_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_342.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_342.w_3_0 ctx) < 20054016
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_16_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_342.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_342.w_5_0 ctx) < 20054016
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_16_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_18_0 ctx)
  h3_0_8 : 0 ≤ (sequential_generatedRoot_342.w_3_0 ctx)
  h3_0_9 : (sequential_generatedRoot_342.w_3_0 ctx) < 20054016
  h21_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_20_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_21_0 ctx)
  h5_0_8 : 0 ≤ (sequential_generatedRoot_342.w_5_0 ctx)
  h5_0_9 : (sequential_generatedRoot_342.w_5_0 ctx) < 20054016
  h22_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_20_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_22_0 ctx)

theorem sequential_generatedRoot_342.facts_0  {ctx : sequential_generatedRoot_342Ctx} {outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit}
    (h : sequential_generatedRoot_342.constraints_0 i_0 (sequential_generatedRoot_342.w_2_0 ctx) (sequential_generatedRoot_342.w_8_0 ctx) (sequential_generatedRoot_342.w_12_0 ctx) (sequential_generatedRoot_342.w_16_0 ctx) (sequential_generatedRoot_342.w_20_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_4_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_6_0 ctx) (sequential_generatedRoot_342.w_9_0 ctx) (sequential_generatedRoot_342.w_10_0 ctx) (sequential_generatedRoot_342.w_13_0 ctx) (sequential_generatedRoot_342.w_14_0 ctx) (sequential_generatedRoot_342.w_17_0 ctx) (sequential_generatedRoot_342.w_18_0 ctx) (sequential_generatedRoot_342.w_21_0 ctx) (sequential_generatedRoot_342.w_22_0 ctx)) :
    sequential_generatedRoot_342.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_342.Facts_1  (ctx : sequential_generatedRoot_342Ctx) (outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit) : Prop where
  h3_0_10 : 0 ≤ (sequential_generatedRoot_342.w_3_0 ctx)
  h3_0_11 : (sequential_generatedRoot_342.w_3_0 ctx) < 20054016
  h25_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_24_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_25_0 ctx)
  h5_0_10 : 0 ≤ (sequential_generatedRoot_342.w_5_0 ctx)
  h5_0_11 : (sequential_generatedRoot_342.w_5_0 ctx) < 20054016
  h26_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_342.w_24_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_26_0 ctx)
  t38 : 0 ≤ (65534)
  h28_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_145 backend hashModel tape (path ++ [28, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((sequential_generatedRoot_342.w_2_0 ctx), ((sequential_generatedRoot_342.w_8_0 ctx), ((sequential_generatedRoot_342.w_12_0 ctx), ((sequential_generatedRoot_342.w_16_0 ctx), ((sequential_generatedRoot_342.w_20_0 ctx), ((sequential_generatedRoot_342.w_24_0 ctx), ())))))))))))) next) (Int.toNat (65534)) ((sequential_generatedRoot_342.w_7_0 ctx), ((sequential_generatedRoot_342.w_11_0 ctx), ((sequential_generatedRoot_342.w_15_0 ctx), ((sequential_generatedRoot_342.w_19_0 ctx), ((sequential_generatedRoot_342.w_23_0 ctx), ((sequential_generatedRoot_342.w_27_0 ctx), ())))))) ((sequential_generatedRoot_342.w_28_0 ctx), ((sequential_generatedRoot_342.w_28_1 ctx), ((sequential_generatedRoot_342.w_28_2 ctx), ((sequential_generatedRoot_342.w_28_3 ctx), ((sequential_generatedRoot_342.w_28_4 ctx), ((sequential_generatedRoot_342.w_28_5 ctx), ()))))))
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_342.w_1_0 ctx) (sequential_generatedRoot_342.w_28_0 ctx) (sequential_generatedRoot_342.w_29_0 ctx)
  t41 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t42 : ((1 + (Int.ofNat i_0))) < 306
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_342.w_32_0 ctx) (sequential_generatedRoot_342.w_28_1 ctx) (sequential_generatedRoot_342.w_33_0 ctx)
  t44 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t45 : ((1 + (Int.ofNat i_0))) < 306
  h37_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_342.w_36_0 ctx) (sequential_generatedRoot_342.w_28_2 ctx) (sequential_generatedRoot_342.w_37_0 ctx)
  t47 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t48 : ((1 + (Int.ofNat i_0))) < 306
  h41_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_342.w_40_0 ctx) (sequential_generatedRoot_342.w_28_3 ctx) (sequential_generatedRoot_342.w_41_0 ctx)
  t50 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t51 : ((1 + (Int.ofNat i_0))) < 306
  h45_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_342.w_44_0 ctx) (sequential_generatedRoot_342.w_28_4 ctx) (sequential_generatedRoot_342.w_45_0 ctx)
  t53 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t54 : ((1 + (Int.ofNat i_0))) < 306
  h49_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_342.w_48_0 ctx) (sequential_generatedRoot_342.w_28_5 ctx) (sequential_generatedRoot_342.w_49_0 ctx)
  hout : outputs = sequential_generatedRoot_342.outs ctx

theorem sequential_generatedRoot_342.facts_1  {ctx : sequential_generatedRoot_342Ctx} {outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit}
    (h : sequential_generatedRoot_342.constraints_1 backend hashModel tape path params i_0 (sequential_generatedRoot_342.w_2_0 ctx) (sequential_generatedRoot_342.w_8_0 ctx) (sequential_generatedRoot_342.w_12_0 ctx) (sequential_generatedRoot_342.w_16_0 ctx) (sequential_generatedRoot_342.w_20_0 ctx) (sequential_generatedRoot_342.w_24_0 ctx) (sequential_generatedRoot_342.w_1_0 ctx) (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_7_0 ctx) (sequential_generatedRoot_342.w_11_0 ctx) (sequential_generatedRoot_342.w_15_0 ctx) (sequential_generatedRoot_342.w_19_0 ctx) (sequential_generatedRoot_342.w_23_0 ctx) (sequential_generatedRoot_342.w_25_0 ctx) (sequential_generatedRoot_342.w_26_0 ctx) (sequential_generatedRoot_342.w_27_0 ctx) (sequential_generatedRoot_342.w_28_0 ctx) (sequential_generatedRoot_342.w_28_1 ctx) (sequential_generatedRoot_342.w_28_2 ctx) (sequential_generatedRoot_342.w_28_3 ctx) (sequential_generatedRoot_342.w_28_4 ctx) (sequential_generatedRoot_342.w_28_5 ctx) (sequential_generatedRoot_342.w_29_0 ctx) (sequential_generatedRoot_342.w_30_0 ctx) (sequential_generatedRoot_342.w_32_0 ctx) (sequential_generatedRoot_342.w_33_0 ctx) (sequential_generatedRoot_342.w_34_0 ctx) (sequential_generatedRoot_342.w_36_0 ctx) (sequential_generatedRoot_342.w_37_0 ctx) (sequential_generatedRoot_342.w_38_0 ctx) (sequential_generatedRoot_342.w_40_0 ctx) (sequential_generatedRoot_342.w_41_0 ctx) (sequential_generatedRoot_342.w_42_0 ctx) (sequential_generatedRoot_342.w_44_0 ctx) (sequential_generatedRoot_342.w_45_0 ctx) (sequential_generatedRoot_342.w_46_0 ctx) (sequential_generatedRoot_342.w_48_0 ctx) (sequential_generatedRoot_342.w_49_0 ctx) (sequential_generatedRoot_342.w_50_0 ctx) outputs) :
    sequential_generatedRoot_342.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_342.Facts  (ctx : sequential_generatedRoot_342Ctx) (outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit) : Prop where
  c0 : sequential_generatedRoot_342.Facts_0 ctx
  c1 : sequential_generatedRoot_342.Facts_1 ctx outputs

theorem sequential_generatedRoot_342.facts  (ctx : sequential_generatedRoot_342Ctx) {outputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit}
    (h : sequential_generatedRoot_342.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_342.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_342.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_342.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel tape path params)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (generatedRoot.w_1_0 ctx) (0) (generatedRoot.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (generatedRoot.w_3_0 ctx) (0) (generatedRoot.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (generatedRoot.w_5_0 ctx) (0) (generatedRoot.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (generatedRoot.w_7_0 ctx) (0) (generatedRoot.w_8_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (generatedRoot.w_12_0 ctx) (1024) (generatedRoot.w_13_0 ctx)
  h16_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_16_0 ctx)
  h19_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_19_0 ctx)
  h18_0 : MxxRuntime.concatRows (generatedRoot.w_15_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_22_concat_1 ctx)
  h22_0 : MxxRuntime.concatRows (generatedRoot.w_22_concat_1 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (generatedRoot.w_23_0 ctx) (1024) (generatedRoot.w_24_0 ctx)
  h27_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_27_0 ctx)
  h30_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_30_0 ctx)
  h29_0 : MxxRuntime.concatRows (generatedRoot.w_26_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_33_concat_1 ctx)
  h33_0 : MxxRuntime.concatRows (generatedRoot.w_33_concat_1 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx)
  h34_0 : MxxRuntime.concatColumns (generatedRoot.w_22_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx)
  h35_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_35_0 ctx)
  h36_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_36_0 ctx)
  h37_0 : MxxRuntime.concatRows (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx)
  h38_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx)
  h39_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx)
  t20 : 0 ≤ 0
  t21 : 0 < 1
  t22 : 1 ≤ 3
  t23 : 0 ≤ 0
  t24 : 0 < 270
  t25 : 270 ≤ 270
  h41_0 : MxxRuntime.sliceMatrix (generatedRoot.w_40_0 ctx) 0 1 0 270 (generatedRoot.w_41_0 ctx)
  h44_0 : MxxRuntime.familyGetStatic (generatedRoot.w_43_0 ctx) (1024) (generatedRoot.w_44_0 ctx)
  t28 : 0 ≤ 1
  t29 : 1 < 2
  t30 : 2 ≤ 3
  t31 : 0 ≤ 0

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_22_concat_1 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_concat_1 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_44_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  t32 : 0 < 270
  t33 : 270 ≤ 270
  h49_0 : MxxRuntime.sliceMatrix (generatedRoot.w_40_0 ctx) 1 2 0 270 (generatedRoot.w_49_0 ctx)
  t35 : 0 ≤ 2
  t36 : 2 < 3
  t37 : 3 ≤ 3
  t38 : 0 ≤ 0
  t39 : 0 < 270
  t40 : 270 ≤ 270
  h50_0 : MxxRuntime.sliceMatrix (generatedRoot.w_40_0 ctx) 2 3 0 270 (generatedRoot.w_50_0 ctx)
  h49_0_1 : MxxRuntime.concatRows (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_concat_1 ctx)
  h51_0 : MxxRuntime.concatRows (generatedRoot.w_51_concat_1 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx)
  h52_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_52_0 ctx)
  h53_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_53_0 ctx)
  h54_0 : MxxRuntime.concatRows (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx)
  h55_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx)
  h56_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx)
  t49 : 0 ≤ 0
  t50 : 0 < 1
  t51 : 1 ≤ 3
  t52 : 0 ≤ 0
  t53 : 0 < 270
  t54 : 270 ≤ 270
  h58_0 : MxxRuntime.sliceMatrix (generatedRoot.w_57_0 ctx) 0 1 0 270 (generatedRoot.w_58_0 ctx)
  t56 : 0 ≤ 1
  t57 : 1 < 2
  t58 : 2 ≤ 3
  t59 : 0 ≤ 0
  t60 : 0 < 270
  t61 : 270 ≤ 270
  h62_0 : MxxRuntime.sliceMatrix (generatedRoot.w_57_0 ctx) 1 2 0 270 (generatedRoot.w_62_0 ctx)
  t63 : 0 ≤ 2

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_10_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_51_concat_1 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_62_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
  t64 : 2 < 3
  t65 : 3 ≤ 3
  t66 : 0 ≤ 0
  t67 : 0 < 270
  t68 : 270 ≤ 270
  h63_0 : MxxRuntime.sliceMatrix (generatedRoot.w_57_0 ctx) 2 3 0 270 (generatedRoot.w_63_0 ctx)
  h62_0_1 : MxxRuntime.concatRows (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_concat_1 ctx)
  h64_0 : MxxRuntime.concatRows (generatedRoot.w_64_concat_1 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx)
  h65_0 : MxxRuntime.concatColumns (generatedRoot.w_51_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_65_0 ctx)
  h67_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 52, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat0), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat1), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_66_0 ctx)) (generatedRoot.w_67_0 ctx)
  h69_0 : MxxRuntime.familyGetStatic (generatedRoot.w_68_0 ctx) (0) (generatedRoot.w_69_0 ctx)
  h70_0 : MxxRuntime.concatRows (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_70_0 ctx)
  h71_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx)
  h72_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx)
  t78 : 0 ≤ 0
  t79 : 0 < 1
  t80 : 1 ≤ 3
  t81 : 0 ≤ 0
  t82 : 0 < 270
  t83 : 270 ≤ 270
  h74_0 : MxxRuntime.sliceMatrix (generatedRoot.w_73_0 ctx) 0 1 0 270 (generatedRoot.w_74_0 ctx)
  t85 : 0 ≤ 1
  t86 : 1 < 2
  t87 : 2 ≤ 3
  t88 : 0 ≤ 0
  t89 : 0 < 270
  t90 : 270 ≤ 270
  h75_0 : MxxRuntime.sliceMatrix (generatedRoot.w_73_0 ctx) 1 2 0 270 (generatedRoot.w_75_0 ctx)
  t92 : 0 ≤ 2
  t93 : 2 < 3
  t94 : 3 ≤ 3
  t95 : 0 ≤ 0

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend hashModel (generatedRoot.w_66_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_64_concat_1 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
  t96 : 0 < 270
  t97 : 270 ≤ 270
  h76_0 : MxxRuntime.sliceMatrix (generatedRoot.w_73_0 ctx) 2 3 0 270 (generatedRoot.w_76_0 ctx)
  h77_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 52, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat0), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat1), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_66_0 ctx)) (generatedRoot.w_77_0 ctx)
  h79_0 : MxxRuntime.familyGetStatic (generatedRoot.w_78_0 ctx) (0) (generatedRoot.w_79_0 ctx)
  h80_0 : MxxRuntime.concatRows (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx)
  h81_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx)
  h82_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx)
  t104 : 0 ≤ 0
  t105 : 0 < 1
  t106 : 1 ≤ 3
  t107 : 0 ≤ 0
  t108 : 0 < 270
  t109 : 270 ≤ 270
  h84_0 : MxxRuntime.sliceMatrix (generatedRoot.w_83_0 ctx) 0 1 0 270 (generatedRoot.w_84_0 ctx)
  t111 : 0 ≤ 1
  t112 : 1 < 2
  t113 : 2 ≤ 3
  t114 : 0 ≤ 0
  t115 : 0 < 270
  t116 : 270 ≤ 270
  h85_0 : MxxRuntime.sliceMatrix (generatedRoot.w_83_0 ctx) 1 2 0 270 (generatedRoot.w_85_0 ctx)
  t118 : 0 ≤ 2
  t119 : 2 < 3
  t120 : 3 ≤ 3
  t121 : 0 ≤ 0
  t122 : 0 < 270
  t123 : 270 ≤ 270
  h86_0 : MxxRuntime.sliceMatrix (generatedRoot.w_83_0 ctx) 2 3 0 270 (generatedRoot.w_86_0 ctx)
  t125 : 0 ≤ (3)
  h87_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_87 backend hashModel tape (path ++ [87, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_74_0 ctx), ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_76_0 ctx), ((generatedRoot.w_84_0 ctx), ((generatedRoot.w_85_0 ctx), ((generatedRoot.w_86_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (3)) ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_8_0 ctx), ())))))))))) ((generatedRoot.w_87_0 ctx), ((generatedRoot.w_87_1 ctx), ((generatedRoot.w_87_2 ctx), ((generatedRoot.w_87_3 ctx), ((generatedRoot.w_87_4 ctx), ((generatedRoot.w_87_5 ctx), ((generatedRoot.w_87_6 ctx), ((generatedRoot.w_87_7 ctx), ((generatedRoot.w_87_8 ctx), ((generatedRoot.w_87_9 ctx), ()))))))))))
  h88_0 : MxxRuntime.familyGetStatic (generatedRoot.w_9_0 ctx) (3) (generatedRoot.w_88_0 ctx)

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_3 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_87_1 ctx) (generatedRoot.w_87_2 ctx) (generatedRoot.w_87_3 ctx) (generatedRoot.w_87_4 ctx) (generatedRoot.w_87_5 ctx) (generatedRoot.w_87_6 ctx) (generatedRoot.w_87_7 ctx) (generatedRoot.w_87_8 ctx) (generatedRoot.w_87_9 ctx) (generatedRoot.w_88_0 ctx)) :
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
  h98_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_74_0 ctx) (generatedRoot.w_98_0 ctx)
  t143 : 0 ≤ (3)
  t144 : (3) < 4
  h99_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_75_0 ctx) (generatedRoot.w_99_0 ctx)
  t146 : 0 ≤ (3)
  t147 : (3) < 4
  h100_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_76_0 ctx) (generatedRoot.w_100_0 ctx)
  h99_0_1 : MxxRuntime.concatRows (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_concat_1 ctx)
  h101_0 : MxxRuntime.concatRows (generatedRoot.w_101_concat_1 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx)
  t151 : 0 ≤ (3)
  t152 : (3) < 4
  h102_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_84_0 ctx) (generatedRoot.w_102_0 ctx)
  t154 : 0 ≤ (3)
  t155 : (3) < 4
  h103_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_85_0 ctx) (generatedRoot.w_103_0 ctx)
  t157 : 0 ≤ (3)
  t158 : (3) < 4
  h104_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (16385), (32769), (49153)])) (generatedRoot.w_86_0 ctx) (generatedRoot.w_104_0 ctx)

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_4 hashModel (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_92_concat_1 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_96_concat_1 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_101_concat_1 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
  h103_0_1 : MxxRuntime.concatRows (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_concat_1 ctx)
  h105_0 : MxxRuntime.concatRows (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx)
  h106_0 : MxxRuntime.concatColumns (generatedRoot.w_101_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_106_0 ctx)
  h107_0 : MxxRuntime.concatRows (generatedRoot.w_87_7 ctx) (generatedRoot.w_87_9 ctx) (generatedRoot.w_107_0 ctx)
  h108_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx)
  h109_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx)
  t166 : 0 ≤ 1
  t167 : 1 < 2
  t168 : 2 ≤ 3
  t169 : 0 ≤ 0
  t170 : 0 < 270
  t171 : 270 ≤ 270
  h111_0 : MxxRuntime.sliceMatrix (generatedRoot.w_110_0 ctx) 1 2 0 270 (generatedRoot.w_111_0 ctx)
  t173 : 0 ≤ 2
  t174 : 2 < 3
  t175 : 3 ≤ 3
  t176 : 0 ≤ 0
  t177 : 0 < 270
  t178 : 270 ≤ 270
  h112_0 : MxxRuntime.sliceMatrix (generatedRoot.w_110_0 ctx) 2 3 0 270 (generatedRoot.w_112_0 ctx)
  h113_0 : MxxRuntime.concatRows (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx)
  h114_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx)
  h115_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx)
  t183 : 0 ≤ 1
  t184 : 1 < 2
  t185 : 2 ≤ 3
  t186 : 0 ≤ 0
  t187 : 0 < 270
  t188 : 270 ≤ 270
  h117_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 1 2 0 270 (generatedRoot.w_117_0 ctx)
  h120_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx)
  h123_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx)

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_5 backend (generatedRoot.w_87_7 ctx) (generatedRoot.w_87_9 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
  h124_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx)
  t193 : 0 ≤ (0)
  t194 : (0) < 306
  h168_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_168_0 ctx)
  h169_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_169_0 ctx)
  h171_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_171_0 ctx)
  h172_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_10_0 ctx)) (generatedRoot.w_172_0 ctx)
  t199 : (2607022080) = 2607022080
  h176_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_167_0 ctx) i) ∧ ((generatedRoot.w_167_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_167_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_167_0 ctx) i) ∧ ((generatedRoot.w_167_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_167_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_167_0 ctx) i) ∧ ((generatedRoot.w_167_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_167_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_176 backend hashModel tape (path ++ [176, i.val]) params i (((generatedRoot.w_167_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_168_0 ctx), ((generatedRoot.w_169_0 ctx), ((generatedRoot.w_170_0 ctx), (w_43_0_gathered, ((generatedRoot.w_171_0 ctx), ((generatedRoot.w_172_0 ctx), ((generatedRoot.w_173_0 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_174_0 ctx) i), (((generatedRoot.w_175_0 ctx) i), ())))))))))))))) (((generatedRoot.w_176_0 ctx) i), (((generatedRoot.w_176_1 ctx) i), (((generatedRoot.w_176_2 ctx) i), (((generatedRoot.w_176_3 ctx) i), (((generatedRoot.w_176_4 ctx) i), (((generatedRoot.w_176_5 ctx) i), ())))))))
  t201 : (20054016) = 20054016
  h177_5 : (∀ i : Fin 20054016, parallel_generatedRoot_177 backend hashModel tape (path ++ [177, i.val]) params i ((generatedRoot.w_176_0 ctx), ((generatedRoot.w_176_1 ctx), ((generatedRoot.w_176_2 ctx), ((generatedRoot.w_176_3 ctx), ((generatedRoot.w_176_4 ctx), ((generatedRoot.w_176_5 ctx), ())))))) (((generatedRoot.w_177_0 ctx) i), (((generatedRoot.w_177_1 ctx) i), (((generatedRoot.w_177_2 ctx) i), (((generatedRoot.w_177_3 ctx) i), (((generatedRoot.w_177_4 ctx) i), (((generatedRoot.w_177_5 ctx) i), ())))))))
  t203 : (2607022080) = 2607022080
  h180_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_166_0 ctx) i) ∧ ((generatedRoot.w_166_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_166_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_166_0 ctx) i) ∧ ((generatedRoot.w_166_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_166_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_166_0 ctx) i) ∧ ((generatedRoot.w_166_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_166_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_180 backend hashModel tape (path ++ [180, i.val]) params i (((generatedRoot.w_166_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_177_0 ctx), ((generatedRoot.w_177_2 ctx), ((generatedRoot.w_177_4 ctx), (w_43_0_gathered, ((generatedRoot.w_177_1 ctx), ((generatedRoot.w_177_3 ctx), ((generatedRoot.w_177_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_178_0 ctx) i), (((generatedRoot.w_179_0 ctx) i), ())))))))))))))) (((generatedRoot.w_180_0 ctx) i), (((generatedRoot.w_180_1 ctx) i), (((generatedRoot.w_180_2 ctx) i), (((generatedRoot.w_180_3 ctx) i), (((generatedRoot.w_180_4 ctx) i), (((generatedRoot.w_180_5 ctx) i), ())))))))
  t205 : (20054016) = 20054016
  h181_5 : (∀ i : Fin 20054016, parallel_generatedRoot_181 backend hashModel tape (path ++ [181, i.val]) params i ((generatedRoot.w_180_0 ctx), ((generatedRoot.w_180_1 ctx), ((generatedRoot.w_180_2 ctx), ((generatedRoot.w_180_3 ctx), ((generatedRoot.w_180_4 ctx), ((generatedRoot.w_180_5 ctx), ())))))) (((generatedRoot.w_181_0 ctx) i), (((generatedRoot.w_181_1 ctx) i), (((generatedRoot.w_181_2 ctx) i), (((generatedRoot.w_181_3 ctx) i), (((generatedRoot.w_181_4 ctx) i), (((generatedRoot.w_181_5 ctx) i), ())))))))
  t207 : (2607022080) = 2607022080
  h184_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_165_0 ctx) i) ∧ ((generatedRoot.w_165_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_165_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_165_0 ctx) i) ∧ ((generatedRoot.w_165_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_165_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_165_0 ctx) i) ∧ ((generatedRoot.w_165_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_165_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_184 backend hashModel tape (path ++ [184, i.val]) params i (((generatedRoot.w_165_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_181_0 ctx), ((generatedRoot.w_181_2 ctx), ((generatedRoot.w_181_4 ctx), (w_43_0_gathered, ((generatedRoot.w_181_1 ctx), ((generatedRoot.w_181_3 ctx), ((generatedRoot.w_181_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_182_0 ctx) i), (((generatedRoot.w_183_0 ctx) i), ())))))))))))))) (((generatedRoot.w_184_0 ctx) i), (((generatedRoot.w_184_1 ctx) i), (((generatedRoot.w_184_2 ctx) i), (((generatedRoot.w_184_3 ctx) i), (((generatedRoot.w_184_4 ctx) i), (((generatedRoot.w_184_5 ctx) i), ())))))))
  t209 : (20054016) = 20054016
  h185_5 : (∀ i : Fin 20054016, parallel_generatedRoot_185 backend hashModel tape (path ++ [185, i.val]) params i ((generatedRoot.w_184_0 ctx), ((generatedRoot.w_184_1 ctx), ((generatedRoot.w_184_2 ctx), ((generatedRoot.w_184_3 ctx), ((generatedRoot.w_184_4 ctx), ((generatedRoot.w_184_5 ctx), ())))))) (((generatedRoot.w_185_0 ctx) i), (((generatedRoot.w_185_1 ctx) i), (((generatedRoot.w_185_2 ctx) i), (((generatedRoot.w_185_3 ctx) i), (((generatedRoot.w_185_4 ctx) i), (((generatedRoot.w_185_5 ctx) i), ())))))))
  t211 : (2607022080) = 2607022080
  h188_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_164_0 ctx) i) ∧ ((generatedRoot.w_164_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_164_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_164_0 ctx) i) ∧ ((generatedRoot.w_164_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_164_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_164_0 ctx) i) ∧ ((generatedRoot.w_164_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_164_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_188 backend hashModel tape (path ++ [188, i.val]) params i (((generatedRoot.w_164_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_185_0 ctx), ((generatedRoot.w_185_2 ctx), ((generatedRoot.w_185_4 ctx), (w_43_0_gathered, ((generatedRoot.w_185_1 ctx), ((generatedRoot.w_185_3 ctx), ((generatedRoot.w_185_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_186_0 ctx) i), (((generatedRoot.w_187_0 ctx) i), ())))))))))))))) (((generatedRoot.w_188_0 ctx) i), (((generatedRoot.w_188_1 ctx) i), (((generatedRoot.w_188_2 ctx) i), (((generatedRoot.w_188_3 ctx) i), (((generatedRoot.w_188_4 ctx) i), (((generatedRoot.w_188_5 ctx) i), ())))))))
  t213 : (20054016) = 20054016
  h189_5 : (∀ i : Fin 20054016, parallel_generatedRoot_189 backend hashModel tape (path ++ [189, i.val]) params i ((generatedRoot.w_188_0 ctx), ((generatedRoot.w_188_1 ctx), ((generatedRoot.w_188_2 ctx), ((generatedRoot.w_188_3 ctx), ((generatedRoot.w_188_4 ctx), ((generatedRoot.w_188_5 ctx), ())))))) (((generatedRoot.w_189_0 ctx) i), (((generatedRoot.w_189_1 ctx) i), (((generatedRoot.w_189_2 ctx) i), (((generatedRoot.w_189_3 ctx) i), (((generatedRoot.w_189_4 ctx) i), (((generatedRoot.w_189_5 ctx) i), ())))))))
  t215 : (2607022080) = 2607022080
  h192_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_163_0 ctx) i) ∧ ((generatedRoot.w_163_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_163_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_163_0 ctx) i) ∧ ((generatedRoot.w_163_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_163_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_163_0 ctx) i) ∧ ((generatedRoot.w_163_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_163_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_192 backend hashModel tape (path ++ [192, i.val]) params i (((generatedRoot.w_163_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_189_0 ctx), ((generatedRoot.w_189_2 ctx), ((generatedRoot.w_189_4 ctx), (w_43_0_gathered, ((generatedRoot.w_189_1 ctx), ((generatedRoot.w_189_3 ctx), ((generatedRoot.w_189_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_190_0 ctx) i), (((generatedRoot.w_191_0 ctx) i), ())))))))))))))) (((generatedRoot.w_192_0 ctx) i), (((generatedRoot.w_192_1 ctx) i), (((generatedRoot.w_192_2 ctx) i), (((generatedRoot.w_192_3 ctx) i), (((generatedRoot.w_192_4 ctx) i), (((generatedRoot.w_192_5 ctx) i), ())))))))
  t217 : (20054016) = 20054016
  h193_5 : (∀ i : Fin 20054016, parallel_generatedRoot_193 backend hashModel tape (path ++ [193, i.val]) params i ((generatedRoot.w_192_0 ctx), ((generatedRoot.w_192_1 ctx), ((generatedRoot.w_192_2 ctx), ((generatedRoot.w_192_3 ctx), ((generatedRoot.w_192_4 ctx), ((generatedRoot.w_192_5 ctx), ())))))) (((generatedRoot.w_193_0 ctx) i), (((generatedRoot.w_193_1 ctx) i), (((generatedRoot.w_193_2 ctx) i), (((generatedRoot.w_193_3 ctx) i), (((generatedRoot.w_193_4 ctx) i), (((generatedRoot.w_193_5 ctx) i), ())))))))
  t219 : (2607022080) = 2607022080
  h196_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_162_0 ctx) i) ∧ ((generatedRoot.w_162_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_162_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_162_0 ctx) i) ∧ ((generatedRoot.w_162_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_162_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_162_0 ctx) i) ∧ ((generatedRoot.w_162_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_162_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_196 backend hashModel tape (path ++ [196, i.val]) params i (((generatedRoot.w_162_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_193_0 ctx), ((generatedRoot.w_193_2 ctx), ((generatedRoot.w_193_4 ctx), (w_43_0_gathered, ((generatedRoot.w_193_1 ctx), ((generatedRoot.w_193_3 ctx), ((generatedRoot.w_193_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_194_0 ctx) i), (((generatedRoot.w_195_0 ctx) i), ())))))))))))))) (((generatedRoot.w_196_0 ctx) i), (((generatedRoot.w_196_1 ctx) i), (((generatedRoot.w_196_2 ctx) i), (((generatedRoot.w_196_3 ctx) i), (((generatedRoot.w_196_4 ctx) i), (((generatedRoot.w_196_5 ctx) i), ())))))))
  t221 : (20054016) = 20054016
  h197_5 : (∀ i : Fin 20054016, parallel_generatedRoot_197 backend hashModel tape (path ++ [197, i.val]) params i ((generatedRoot.w_196_0 ctx), ((generatedRoot.w_196_1 ctx), ((generatedRoot.w_196_2 ctx), ((generatedRoot.w_196_3 ctx), ((generatedRoot.w_196_4 ctx), ((generatedRoot.w_196_5 ctx), ())))))) (((generatedRoot.w_197_0 ctx) i), (((generatedRoot.w_197_1 ctx) i), (((generatedRoot.w_197_2 ctx) i), (((generatedRoot.w_197_3 ctx) i), (((generatedRoot.w_197_4 ctx) i), (((generatedRoot.w_197_5 ctx) i), ())))))))
  t223 : (2607022080) = 2607022080

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_6 backend hashModel tape path params (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_176_1 ctx) (generatedRoot.w_176_2 ctx) (generatedRoot.w_176_3 ctx) (generatedRoot.w_176_4 ctx) (generatedRoot.w_176_5 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_177_1 ctx) (generatedRoot.w_177_2 ctx) (generatedRoot.w_177_3 ctx) (generatedRoot.w_177_4 ctx) (generatedRoot.w_177_5 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_181_1 ctx) (generatedRoot.w_181_2 ctx) (generatedRoot.w_181_3 ctx) (generatedRoot.w_181_4 ctx) (generatedRoot.w_181_5 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_184_1 ctx) (generatedRoot.w_184_2 ctx) (generatedRoot.w_184_3 ctx) (generatedRoot.w_184_4 ctx) (generatedRoot.w_184_5 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_185_1 ctx) (generatedRoot.w_185_2 ctx) (generatedRoot.w_185_3 ctx) (generatedRoot.w_185_4 ctx) (generatedRoot.w_185_5 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_188_1 ctx) (generatedRoot.w_188_2 ctx) (generatedRoot.w_188_3 ctx) (generatedRoot.w_188_4 ctx) (generatedRoot.w_188_5 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_189_1 ctx) (generatedRoot.w_189_2 ctx) (generatedRoot.w_189_3 ctx) (generatedRoot.w_189_4 ctx) (generatedRoot.w_189_5 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_192_1 ctx) (generatedRoot.w_192_2 ctx) (generatedRoot.w_192_3 ctx) (generatedRoot.w_192_4 ctx) (generatedRoot.w_192_5 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_193_1 ctx) (generatedRoot.w_193_2 ctx) (generatedRoot.w_193_3 ctx) (generatedRoot.w_193_4 ctx) (generatedRoot.w_193_5 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_196_1 ctx) (generatedRoot.w_196_2 ctx) (generatedRoot.w_196_3 ctx) (generatedRoot.w_196_4 ctx) (generatedRoot.w_196_5 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_197_1 ctx) (generatedRoot.w_197_2 ctx) (generatedRoot.w_197_3 ctx) (generatedRoot.w_197_4 ctx) (generatedRoot.w_197_5 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) : Prop where
  h200_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_161_0 ctx) i) ∧ ((generatedRoot.w_161_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_161_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_161_0 ctx) i) ∧ ((generatedRoot.w_161_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_161_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_161_0 ctx) i) ∧ ((generatedRoot.w_161_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_161_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_200 backend hashModel tape (path ++ [200, i.val]) params i (((generatedRoot.w_161_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_197_0 ctx), ((generatedRoot.w_197_2 ctx), ((generatedRoot.w_197_4 ctx), (w_43_0_gathered, ((generatedRoot.w_197_1 ctx), ((generatedRoot.w_197_3 ctx), ((generatedRoot.w_197_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_198_0 ctx) i), (((generatedRoot.w_199_0 ctx) i), ())))))))))))))) (((generatedRoot.w_200_0 ctx) i), (((generatedRoot.w_200_1 ctx) i), (((generatedRoot.w_200_2 ctx) i), (((generatedRoot.w_200_3 ctx) i), (((generatedRoot.w_200_4 ctx) i), (((generatedRoot.w_200_5 ctx) i), ())))))))
  t225 : (20054016) = 20054016
  h201_5 : (∀ i : Fin 20054016, parallel_generatedRoot_201 backend hashModel tape (path ++ [201, i.val]) params i ((generatedRoot.w_200_0 ctx), ((generatedRoot.w_200_1 ctx), ((generatedRoot.w_200_2 ctx), ((generatedRoot.w_200_3 ctx), ((generatedRoot.w_200_4 ctx), ((generatedRoot.w_200_5 ctx), ())))))) (((generatedRoot.w_201_0 ctx) i), (((generatedRoot.w_201_1 ctx) i), (((generatedRoot.w_201_2 ctx) i), (((generatedRoot.w_201_3 ctx) i), (((generatedRoot.w_201_4 ctx) i), (((generatedRoot.w_201_5 ctx) i), ())))))))
  t227 : (2607022080) = 2607022080
  h204_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_160_0 ctx) i) ∧ ((generatedRoot.w_160_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_160_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_160_0 ctx) i) ∧ ((generatedRoot.w_160_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_160_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_160_0 ctx) i) ∧ ((generatedRoot.w_160_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_160_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_204 backend hashModel tape (path ++ [204, i.val]) params i (((generatedRoot.w_160_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_201_0 ctx), ((generatedRoot.w_201_2 ctx), ((generatedRoot.w_201_4 ctx), (w_43_0_gathered, ((generatedRoot.w_201_1 ctx), ((generatedRoot.w_201_3 ctx), ((generatedRoot.w_201_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_202_0 ctx) i), (((generatedRoot.w_203_0 ctx) i), ())))))))))))))) (((generatedRoot.w_204_0 ctx) i), (((generatedRoot.w_204_1 ctx) i), (((generatedRoot.w_204_2 ctx) i), (((generatedRoot.w_204_3 ctx) i), (((generatedRoot.w_204_4 ctx) i), (((generatedRoot.w_204_5 ctx) i), ())))))))
  t229 : (20054016) = 20054016
  h205_5 : (∀ i : Fin 20054016, parallel_generatedRoot_205 backend hashModel tape (path ++ [205, i.val]) params i ((generatedRoot.w_204_0 ctx), ((generatedRoot.w_204_1 ctx), ((generatedRoot.w_204_2 ctx), ((generatedRoot.w_204_3 ctx), ((generatedRoot.w_204_4 ctx), ((generatedRoot.w_204_5 ctx), ())))))) (((generatedRoot.w_205_0 ctx) i), (((generatedRoot.w_205_1 ctx) i), (((generatedRoot.w_205_2 ctx) i), (((generatedRoot.w_205_3 ctx) i), (((generatedRoot.w_205_4 ctx) i), (((generatedRoot.w_205_5 ctx) i), ())))))))
  t231 : (2607022080) = 2607022080
  h208_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_159_0 ctx) i) ∧ ((generatedRoot.w_159_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_159_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_159_0 ctx) i) ∧ ((generatedRoot.w_159_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_159_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_159_0 ctx) i) ∧ ((generatedRoot.w_159_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_159_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_208 backend hashModel tape (path ++ [208, i.val]) params i (((generatedRoot.w_159_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_205_0 ctx), ((generatedRoot.w_205_2 ctx), ((generatedRoot.w_205_4 ctx), (w_43_0_gathered, ((generatedRoot.w_205_1 ctx), ((generatedRoot.w_205_3 ctx), ((generatedRoot.w_205_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_206_0 ctx) i), (((generatedRoot.w_207_0 ctx) i), ())))))))))))))) (((generatedRoot.w_208_0 ctx) i), (((generatedRoot.w_208_1 ctx) i), (((generatedRoot.w_208_2 ctx) i), (((generatedRoot.w_208_3 ctx) i), (((generatedRoot.w_208_4 ctx) i), (((generatedRoot.w_208_5 ctx) i), ())))))))
  t233 : (20054016) = 20054016
  h209_5 : (∀ i : Fin 20054016, parallel_generatedRoot_209 backend hashModel tape (path ++ [209, i.val]) params i ((generatedRoot.w_208_0 ctx), ((generatedRoot.w_208_1 ctx), ((generatedRoot.w_208_2 ctx), ((generatedRoot.w_208_3 ctx), ((generatedRoot.w_208_4 ctx), ((generatedRoot.w_208_5 ctx), ())))))) (((generatedRoot.w_209_0 ctx) i), (((generatedRoot.w_209_1 ctx) i), (((generatedRoot.w_209_2 ctx) i), (((generatedRoot.w_209_3 ctx) i), (((generatedRoot.w_209_4 ctx) i), (((generatedRoot.w_209_5 ctx) i), ())))))))
  t235 : (2607022080) = 2607022080
  h212_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_158_0 ctx) i) ∧ ((generatedRoot.w_158_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_158_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_158_0 ctx) i) ∧ ((generatedRoot.w_158_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_158_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_158_0 ctx) i) ∧ ((generatedRoot.w_158_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_158_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_212 backend hashModel tape (path ++ [212, i.val]) params i (((generatedRoot.w_158_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_209_0 ctx), ((generatedRoot.w_209_2 ctx), ((generatedRoot.w_209_4 ctx), (w_43_0_gathered, ((generatedRoot.w_209_1 ctx), ((generatedRoot.w_209_3 ctx), ((generatedRoot.w_209_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_210_0 ctx) i), (((generatedRoot.w_211_0 ctx) i), ())))))))))))))) (((generatedRoot.w_212_0 ctx) i), (((generatedRoot.w_212_1 ctx) i), (((generatedRoot.w_212_2 ctx) i), (((generatedRoot.w_212_3 ctx) i), (((generatedRoot.w_212_4 ctx) i), (((generatedRoot.w_212_5 ctx) i), ())))))))
  t237 : (20054016) = 20054016
  h213_5 : (∀ i : Fin 20054016, parallel_generatedRoot_213 backend hashModel tape (path ++ [213, i.val]) params i ((generatedRoot.w_212_0 ctx), ((generatedRoot.w_212_1 ctx), ((generatedRoot.w_212_2 ctx), ((generatedRoot.w_212_3 ctx), ((generatedRoot.w_212_4 ctx), ((generatedRoot.w_212_5 ctx), ())))))) (((generatedRoot.w_213_0 ctx) i), (((generatedRoot.w_213_1 ctx) i), (((generatedRoot.w_213_2 ctx) i), (((generatedRoot.w_213_3 ctx) i), (((generatedRoot.w_213_4 ctx) i), (((generatedRoot.w_213_5 ctx) i), ())))))))
  t239 : (2607022080) = 2607022080
  h216_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_157_0 ctx) i) ∧ ((generatedRoot.w_157_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_157_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_157_0 ctx) i) ∧ ((generatedRoot.w_157_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_157_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_157_0 ctx) i) ∧ ((generatedRoot.w_157_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_157_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_216 backend hashModel tape (path ++ [216, i.val]) params i (((generatedRoot.w_157_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_213_0 ctx), ((generatedRoot.w_213_2 ctx), ((generatedRoot.w_213_4 ctx), (w_43_0_gathered, ((generatedRoot.w_213_1 ctx), ((generatedRoot.w_213_3 ctx), ((generatedRoot.w_213_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_214_0 ctx) i), (((generatedRoot.w_215_0 ctx) i), ())))))))))))))) (((generatedRoot.w_216_0 ctx) i), (((generatedRoot.w_216_1 ctx) i), (((generatedRoot.w_216_2 ctx) i), (((generatedRoot.w_216_3 ctx) i), (((generatedRoot.w_216_4 ctx) i), (((generatedRoot.w_216_5 ctx) i), ())))))))
  t241 : (20054016) = 20054016
  h217_5 : (∀ i : Fin 20054016, parallel_generatedRoot_217 backend hashModel tape (path ++ [217, i.val]) params i ((generatedRoot.w_216_0 ctx), ((generatedRoot.w_216_1 ctx), ((generatedRoot.w_216_2 ctx), ((generatedRoot.w_216_3 ctx), ((generatedRoot.w_216_4 ctx), ((generatedRoot.w_216_5 ctx), ())))))) (((generatedRoot.w_217_0 ctx) i), (((generatedRoot.w_217_1 ctx) i), (((generatedRoot.w_217_2 ctx) i), (((generatedRoot.w_217_3 ctx) i), (((generatedRoot.w_217_4 ctx) i), (((generatedRoot.w_217_5 ctx) i), ())))))))
  t243 : (2607022080) = 2607022080
  h220_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_156_0 ctx) i) ∧ ((generatedRoot.w_156_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_156_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_156_0 ctx) i) ∧ ((generatedRoot.w_156_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_156_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_156_0 ctx) i) ∧ ((generatedRoot.w_156_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_156_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_220 backend hashModel tape (path ++ [220, i.val]) params i (((generatedRoot.w_156_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_217_0 ctx), ((generatedRoot.w_217_2 ctx), ((generatedRoot.w_217_4 ctx), (w_43_0_gathered, ((generatedRoot.w_217_1 ctx), ((generatedRoot.w_217_3 ctx), ((generatedRoot.w_217_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_218_0 ctx) i), (((generatedRoot.w_219_0 ctx) i), ())))))))))))))) (((generatedRoot.w_220_0 ctx) i), (((generatedRoot.w_220_1 ctx) i), (((generatedRoot.w_220_2 ctx) i), (((generatedRoot.w_220_3 ctx) i), (((generatedRoot.w_220_4 ctx) i), (((generatedRoot.w_220_5 ctx) i), ())))))))
  t245 : (20054016) = 20054016
  h221_5 : (∀ i : Fin 20054016, parallel_generatedRoot_221 backend hashModel tape (path ++ [221, i.val]) params i ((generatedRoot.w_220_0 ctx), ((generatedRoot.w_220_1 ctx), ((generatedRoot.w_220_2 ctx), ((generatedRoot.w_220_3 ctx), ((generatedRoot.w_220_4 ctx), ((generatedRoot.w_220_5 ctx), ())))))) (((generatedRoot.w_221_0 ctx) i), (((generatedRoot.w_221_1 ctx) i), (((generatedRoot.w_221_2 ctx) i), (((generatedRoot.w_221_3 ctx) i), (((generatedRoot.w_221_4 ctx) i), (((generatedRoot.w_221_5 ctx) i), ())))))))
  t247 : (2607022080) = 2607022080
  h224_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_155_0 ctx) i) ∧ ((generatedRoot.w_155_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_155_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_155_0 ctx) i) ∧ ((generatedRoot.w_155_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_155_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_155_0 ctx) i) ∧ ((generatedRoot.w_155_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_155_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_224 backend hashModel tape (path ++ [224, i.val]) params i (((generatedRoot.w_155_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_221_0 ctx), ((generatedRoot.w_221_2 ctx), ((generatedRoot.w_221_4 ctx), (w_43_0_gathered, ((generatedRoot.w_221_1 ctx), ((generatedRoot.w_221_3 ctx), ((generatedRoot.w_221_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_222_0 ctx) i), (((generatedRoot.w_223_0 ctx) i), ())))))))))))))) (((generatedRoot.w_224_0 ctx) i), (((generatedRoot.w_224_1 ctx) i), (((generatedRoot.w_224_2 ctx) i), (((generatedRoot.w_224_3 ctx) i), (((generatedRoot.w_224_4 ctx) i), (((generatedRoot.w_224_5 ctx) i), ())))))))
  t249 : (20054016) = 20054016
  h225_5 : (∀ i : Fin 20054016, parallel_generatedRoot_225 backend hashModel tape (path ++ [225, i.val]) params i ((generatedRoot.w_224_0 ctx), ((generatedRoot.w_224_1 ctx), ((generatedRoot.w_224_2 ctx), ((generatedRoot.w_224_3 ctx), ((generatedRoot.w_224_4 ctx), ((generatedRoot.w_224_5 ctx), ())))))) (((generatedRoot.w_225_0 ctx) i), (((generatedRoot.w_225_1 ctx) i), (((generatedRoot.w_225_2 ctx) i), (((generatedRoot.w_225_3 ctx) i), (((generatedRoot.w_225_4 ctx) i), (((generatedRoot.w_225_5 ctx) i), ())))))))
  t251 : (2607022080) = 2607022080
  h228_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_154_0 ctx) i) ∧ ((generatedRoot.w_154_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_154_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_154_0 ctx) i) ∧ ((generatedRoot.w_154_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_154_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_154_0 ctx) i) ∧ ((generatedRoot.w_154_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_154_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_228 backend hashModel tape (path ++ [228, i.val]) params i (((generatedRoot.w_154_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_225_0 ctx), ((generatedRoot.w_225_2 ctx), ((generatedRoot.w_225_4 ctx), (w_43_0_gathered, ((generatedRoot.w_225_1 ctx), ((generatedRoot.w_225_3 ctx), ((generatedRoot.w_225_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_226_0 ctx) i), (((generatedRoot.w_227_0 ctx) i), ())))))))))))))) (((generatedRoot.w_228_0 ctx) i), (((generatedRoot.w_228_1 ctx) i), (((generatedRoot.w_228_2 ctx) i), (((generatedRoot.w_228_3 ctx) i), (((generatedRoot.w_228_4 ctx) i), (((generatedRoot.w_228_5 ctx) i), ())))))))
  t253 : (20054016) = 20054016
  h229_5 : (∀ i : Fin 20054016, parallel_generatedRoot_229 backend hashModel tape (path ++ [229, i.val]) params i ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_1 ctx), ((generatedRoot.w_228_2 ctx), ((generatedRoot.w_228_3 ctx), ((generatedRoot.w_228_4 ctx), ((generatedRoot.w_228_5 ctx), ())))))) (((generatedRoot.w_229_0 ctx) i), (((generatedRoot.w_229_1 ctx) i), (((generatedRoot.w_229_2 ctx) i), (((generatedRoot.w_229_3 ctx) i), (((generatedRoot.w_229_4 ctx) i), (((generatedRoot.w_229_5 ctx) i), ())))))))
  t255 : (2607022080) = 2607022080

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_7 backend hashModel tape path params (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_222_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_197_1 ctx) (generatedRoot.w_197_2 ctx) (generatedRoot.w_197_3 ctx) (generatedRoot.w_197_4 ctx) (generatedRoot.w_197_5 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_201_1 ctx) (generatedRoot.w_201_2 ctx) (generatedRoot.w_201_3 ctx) (generatedRoot.w_201_4 ctx) (generatedRoot.w_201_5 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_204_1 ctx) (generatedRoot.w_204_2 ctx) (generatedRoot.w_204_3 ctx) (generatedRoot.w_204_4 ctx) (generatedRoot.w_204_5 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_205_1 ctx) (generatedRoot.w_205_2 ctx) (generatedRoot.w_205_3 ctx) (generatedRoot.w_205_4 ctx) (generatedRoot.w_205_5 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_208_1 ctx) (generatedRoot.w_208_2 ctx) (generatedRoot.w_208_3 ctx) (generatedRoot.w_208_4 ctx) (generatedRoot.w_208_5 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_209_1 ctx) (generatedRoot.w_209_2 ctx) (generatedRoot.w_209_3 ctx) (generatedRoot.w_209_4 ctx) (generatedRoot.w_209_5 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_212_1 ctx) (generatedRoot.w_212_2 ctx) (generatedRoot.w_212_3 ctx) (generatedRoot.w_212_4 ctx) (generatedRoot.w_212_5 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_213_1 ctx) (generatedRoot.w_213_2 ctx) (generatedRoot.w_213_3 ctx) (generatedRoot.w_213_4 ctx) (generatedRoot.w_213_5 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_216_1 ctx) (generatedRoot.w_216_2 ctx) (generatedRoot.w_216_3 ctx) (generatedRoot.w_216_4 ctx) (generatedRoot.w_216_5 ctx) (generatedRoot.w_217_0 ctx) (generatedRoot.w_217_1 ctx) (generatedRoot.w_217_2 ctx) (generatedRoot.w_217_3 ctx) (generatedRoot.w_217_4 ctx) (generatedRoot.w_217_5 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_221_1 ctx) (generatedRoot.w_221_2 ctx) (generatedRoot.w_221_3 ctx) (generatedRoot.w_221_4 ctx) (generatedRoot.w_221_5 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_224_1 ctx) (generatedRoot.w_224_2 ctx) (generatedRoot.w_224_3 ctx) (generatedRoot.w_224_4 ctx) (generatedRoot.w_224_5 ctx) (generatedRoot.w_225_0 ctx) (generatedRoot.w_225_1 ctx) (generatedRoot.w_225_2 ctx) (generatedRoot.w_225_3 ctx) (generatedRoot.w_225_4 ctx) (generatedRoot.w_225_5 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_228_1 ctx) (generatedRoot.w_228_2 ctx) (generatedRoot.w_228_3 ctx) (generatedRoot.w_228_4 ctx) (generatedRoot.w_228_5 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_229_1 ctx) (generatedRoot.w_229_2 ctx) (generatedRoot.w_229_3 ctx) (generatedRoot.w_229_4 ctx) (generatedRoot.w_229_5 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8  (ctx : generatedRootCtx) : Prop where
  h232_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_153_0 ctx) i) ∧ ((generatedRoot.w_153_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_153_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_153_0 ctx) i) ∧ ((generatedRoot.w_153_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_153_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_153_0 ctx) i) ∧ ((generatedRoot.w_153_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_153_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_232 backend hashModel tape (path ++ [232, i.val]) params i (((generatedRoot.w_153_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_229_2 ctx), ((generatedRoot.w_229_4 ctx), (w_43_0_gathered, ((generatedRoot.w_229_1 ctx), ((generatedRoot.w_229_3 ctx), ((generatedRoot.w_229_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_230_0 ctx) i), (((generatedRoot.w_231_0 ctx) i), ())))))))))))))) (((generatedRoot.w_232_0 ctx) i), (((generatedRoot.w_232_1 ctx) i), (((generatedRoot.w_232_2 ctx) i), (((generatedRoot.w_232_3 ctx) i), (((generatedRoot.w_232_4 ctx) i), (((generatedRoot.w_232_5 ctx) i), ())))))))
  t257 : (20054016) = 20054016
  h233_5 : (∀ i : Fin 20054016, parallel_generatedRoot_233 backend hashModel tape (path ++ [233, i.val]) params i ((generatedRoot.w_232_0 ctx), ((generatedRoot.w_232_1 ctx), ((generatedRoot.w_232_2 ctx), ((generatedRoot.w_232_3 ctx), ((generatedRoot.w_232_4 ctx), ((generatedRoot.w_232_5 ctx), ())))))) (((generatedRoot.w_233_0 ctx) i), (((generatedRoot.w_233_1 ctx) i), (((generatedRoot.w_233_2 ctx) i), (((generatedRoot.w_233_3 ctx) i), (((generatedRoot.w_233_4 ctx) i), (((generatedRoot.w_233_5 ctx) i), ())))))))
  t259 : (2607022080) = 2607022080
  h236_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_152_0 ctx) i) ∧ ((generatedRoot.w_152_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_152_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_152_0 ctx) i) ∧ ((generatedRoot.w_152_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_152_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_152_0 ctx) i) ∧ ((generatedRoot.w_152_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_152_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_236 backend hashModel tape (path ++ [236, i.val]) params i (((generatedRoot.w_152_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_233_0 ctx), ((generatedRoot.w_233_2 ctx), ((generatedRoot.w_233_4 ctx), (w_43_0_gathered, ((generatedRoot.w_233_1 ctx), ((generatedRoot.w_233_3 ctx), ((generatedRoot.w_233_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_234_0 ctx) i), (((generatedRoot.w_235_0 ctx) i), ())))))))))))))) (((generatedRoot.w_236_0 ctx) i), (((generatedRoot.w_236_1 ctx) i), (((generatedRoot.w_236_2 ctx) i), (((generatedRoot.w_236_3 ctx) i), (((generatedRoot.w_236_4 ctx) i), (((generatedRoot.w_236_5 ctx) i), ())))))))
  t261 : (20054016) = 20054016
  h237_5 : (∀ i : Fin 20054016, parallel_generatedRoot_237 backend hashModel tape (path ++ [237, i.val]) params i ((generatedRoot.w_236_0 ctx), ((generatedRoot.w_236_1 ctx), ((generatedRoot.w_236_2 ctx), ((generatedRoot.w_236_3 ctx), ((generatedRoot.w_236_4 ctx), ((generatedRoot.w_236_5 ctx), ())))))) (((generatedRoot.w_237_0 ctx) i), (((generatedRoot.w_237_1 ctx) i), (((generatedRoot.w_237_2 ctx) i), (((generatedRoot.w_237_3 ctx) i), (((generatedRoot.w_237_4 ctx) i), (((generatedRoot.w_237_5 ctx) i), ())))))))
  t263 : (2607022080) = 2607022080
  h240_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_151_0 ctx) i) ∧ ((generatedRoot.w_151_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_151_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_151_0 ctx) i) ∧ ((generatedRoot.w_151_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_151_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_151_0 ctx) i) ∧ ((generatedRoot.w_151_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_151_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_240 backend hashModel tape (path ++ [240, i.val]) params i (((generatedRoot.w_151_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_237_0 ctx), ((generatedRoot.w_237_2 ctx), ((generatedRoot.w_237_4 ctx), (w_43_0_gathered, ((generatedRoot.w_237_1 ctx), ((generatedRoot.w_237_3 ctx), ((generatedRoot.w_237_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_238_0 ctx) i), (((generatedRoot.w_239_0 ctx) i), ())))))))))))))) (((generatedRoot.w_240_0 ctx) i), (((generatedRoot.w_240_1 ctx) i), (((generatedRoot.w_240_2 ctx) i), (((generatedRoot.w_240_3 ctx) i), (((generatedRoot.w_240_4 ctx) i), (((generatedRoot.w_240_5 ctx) i), ())))))))
  t265 : (20054016) = 20054016
  h241_5 : (∀ i : Fin 20054016, parallel_generatedRoot_241 backend hashModel tape (path ++ [241, i.val]) params i ((generatedRoot.w_240_0 ctx), ((generatedRoot.w_240_1 ctx), ((generatedRoot.w_240_2 ctx), ((generatedRoot.w_240_3 ctx), ((generatedRoot.w_240_4 ctx), ((generatedRoot.w_240_5 ctx), ())))))) (((generatedRoot.w_241_0 ctx) i), (((generatedRoot.w_241_1 ctx) i), (((generatedRoot.w_241_2 ctx) i), (((generatedRoot.w_241_3 ctx) i), (((generatedRoot.w_241_4 ctx) i), (((generatedRoot.w_241_5 ctx) i), ())))))))
  t267 : (2607022080) = 2607022080
  h244_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_150_0 ctx) i) ∧ ((generatedRoot.w_150_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_150_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_150_0 ctx) i) ∧ ((generatedRoot.w_150_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_150_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_150_0 ctx) i) ∧ ((generatedRoot.w_150_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_150_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_244 backend hashModel tape (path ++ [244, i.val]) params i (((generatedRoot.w_150_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_241_0 ctx), ((generatedRoot.w_241_2 ctx), ((generatedRoot.w_241_4 ctx), (w_43_0_gathered, ((generatedRoot.w_241_1 ctx), ((generatedRoot.w_241_3 ctx), ((generatedRoot.w_241_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_242_0 ctx) i), (((generatedRoot.w_243_0 ctx) i), ())))))))))))))) (((generatedRoot.w_244_0 ctx) i), (((generatedRoot.w_244_1 ctx) i), (((generatedRoot.w_244_2 ctx) i), (((generatedRoot.w_244_3 ctx) i), (((generatedRoot.w_244_4 ctx) i), (((generatedRoot.w_244_5 ctx) i), ())))))))
  t269 : (20054016) = 20054016
  h245_5 : (∀ i : Fin 20054016, parallel_generatedRoot_245 backend hashModel tape (path ++ [245, i.val]) params i ((generatedRoot.w_244_0 ctx), ((generatedRoot.w_244_1 ctx), ((generatedRoot.w_244_2 ctx), ((generatedRoot.w_244_3 ctx), ((generatedRoot.w_244_4 ctx), ((generatedRoot.w_244_5 ctx), ())))))) (((generatedRoot.w_245_0 ctx) i), (((generatedRoot.w_245_1 ctx) i), (((generatedRoot.w_245_2 ctx) i), (((generatedRoot.w_245_3 ctx) i), (((generatedRoot.w_245_4 ctx) i), (((generatedRoot.w_245_5 ctx) i), ())))))))
  t271 : (2607022080) = 2607022080
  h248_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_149_0 ctx) i) ∧ ((generatedRoot.w_149_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_149_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_149_0 ctx) i) ∧ ((generatedRoot.w_149_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_149_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_149_0 ctx) i) ∧ ((generatedRoot.w_149_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_149_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_248 backend hashModel tape (path ++ [248, i.val]) params i (((generatedRoot.w_149_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_245_0 ctx), ((generatedRoot.w_245_2 ctx), ((generatedRoot.w_245_4 ctx), (w_43_0_gathered, ((generatedRoot.w_245_1 ctx), ((generatedRoot.w_245_3 ctx), ((generatedRoot.w_245_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_246_0 ctx) i), (((generatedRoot.w_247_0 ctx) i), ())))))))))))))) (((generatedRoot.w_248_0 ctx) i), (((generatedRoot.w_248_1 ctx) i), (((generatedRoot.w_248_2 ctx) i), (((generatedRoot.w_248_3 ctx) i), (((generatedRoot.w_248_4 ctx) i), (((generatedRoot.w_248_5 ctx) i), ())))))))
  t273 : (20054016) = 20054016
  h249_5 : (∀ i : Fin 20054016, parallel_generatedRoot_249 backend hashModel tape (path ++ [249, i.val]) params i ((generatedRoot.w_248_0 ctx), ((generatedRoot.w_248_1 ctx), ((generatedRoot.w_248_2 ctx), ((generatedRoot.w_248_3 ctx), ((generatedRoot.w_248_4 ctx), ((generatedRoot.w_248_5 ctx), ())))))) (((generatedRoot.w_249_0 ctx) i), (((generatedRoot.w_249_1 ctx) i), (((generatedRoot.w_249_2 ctx) i), (((generatedRoot.w_249_3 ctx) i), (((generatedRoot.w_249_4 ctx) i), (((generatedRoot.w_249_5 ctx) i), ())))))))
  t275 : (2607022080) = 2607022080
  h252_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_148_0 ctx) i) ∧ ((generatedRoot.w_148_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_148_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_148_0 ctx) i) ∧ ((generatedRoot.w_148_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_148_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_148_0 ctx) i) ∧ ((generatedRoot.w_148_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_148_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_252 backend hashModel tape (path ++ [252, i.val]) params i (((generatedRoot.w_148_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_249_0 ctx), ((generatedRoot.w_249_2 ctx), ((generatedRoot.w_249_4 ctx), (w_43_0_gathered, ((generatedRoot.w_249_1 ctx), ((generatedRoot.w_249_3 ctx), ((generatedRoot.w_249_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_250_0 ctx) i), (((generatedRoot.w_251_0 ctx) i), ())))))))))))))) (((generatedRoot.w_252_0 ctx) i), (((generatedRoot.w_252_1 ctx) i), (((generatedRoot.w_252_2 ctx) i), (((generatedRoot.w_252_3 ctx) i), (((generatedRoot.w_252_4 ctx) i), (((generatedRoot.w_252_5 ctx) i), ())))))))
  t277 : (20054016) = 20054016
  h253_5 : (∀ i : Fin 20054016, parallel_generatedRoot_253 backend hashModel tape (path ++ [253, i.val]) params i ((generatedRoot.w_252_0 ctx), ((generatedRoot.w_252_1 ctx), ((generatedRoot.w_252_2 ctx), ((generatedRoot.w_252_3 ctx), ((generatedRoot.w_252_4 ctx), ((generatedRoot.w_252_5 ctx), ())))))) (((generatedRoot.w_253_0 ctx) i), (((generatedRoot.w_253_1 ctx) i), (((generatedRoot.w_253_2 ctx) i), (((generatedRoot.w_253_3 ctx) i), (((generatedRoot.w_253_4 ctx) i), (((generatedRoot.w_253_5 ctx) i), ())))))))
  t279 : (2607022080) = 2607022080
  h256_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_147_0 ctx) i) ∧ ((generatedRoot.w_147_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_147_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_147_0 ctx) i) ∧ ((generatedRoot.w_147_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_147_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_147_0 ctx) i) ∧ ((generatedRoot.w_147_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_147_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_256 backend hashModel tape (path ++ [256, i.val]) params i (((generatedRoot.w_147_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_253_0 ctx), ((generatedRoot.w_253_2 ctx), ((generatedRoot.w_253_4 ctx), (w_43_0_gathered, ((generatedRoot.w_253_1 ctx), ((generatedRoot.w_253_3 ctx), ((generatedRoot.w_253_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_254_0 ctx) i), (((generatedRoot.w_255_0 ctx) i), ())))))))))))))) (((generatedRoot.w_256_0 ctx) i), (((generatedRoot.w_256_1 ctx) i), (((generatedRoot.w_256_2 ctx) i), (((generatedRoot.w_256_3 ctx) i), (((generatedRoot.w_256_4 ctx) i), (((generatedRoot.w_256_5 ctx) i), ())))))))
  t281 : (20054016) = 20054016
  h257_5 : (∀ i : Fin 20054016, parallel_generatedRoot_257 backend hashModel tape (path ++ [257, i.val]) params i ((generatedRoot.w_256_0 ctx), ((generatedRoot.w_256_1 ctx), ((generatedRoot.w_256_2 ctx), ((generatedRoot.w_256_3 ctx), ((generatedRoot.w_256_4 ctx), ((generatedRoot.w_256_5 ctx), ())))))) (((generatedRoot.w_257_0 ctx) i), (((generatedRoot.w_257_1 ctx) i), (((generatedRoot.w_257_2 ctx) i), (((generatedRoot.w_257_3 ctx) i), (((generatedRoot.w_257_4 ctx) i), (((generatedRoot.w_257_5 ctx) i), ())))))))
  t283 : (2607022080) = 2607022080
  h260_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_146_0 ctx) i) ∧ ((generatedRoot.w_146_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_146_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_146_0 ctx) i) ∧ ((generatedRoot.w_146_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_146_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_146_0 ctx) i) ∧ ((generatedRoot.w_146_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_146_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_260 backend hashModel tape (path ++ [260, i.val]) params i (((generatedRoot.w_146_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_257_0 ctx), ((generatedRoot.w_257_2 ctx), ((generatedRoot.w_257_4 ctx), (w_43_0_gathered, ((generatedRoot.w_257_1 ctx), ((generatedRoot.w_257_3 ctx), ((generatedRoot.w_257_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_258_0 ctx) i), (((generatedRoot.w_259_0 ctx) i), ())))))))))))))) (((generatedRoot.w_260_0 ctx) i), (((generatedRoot.w_260_1 ctx) i), (((generatedRoot.w_260_2 ctx) i), (((generatedRoot.w_260_3 ctx) i), (((generatedRoot.w_260_4 ctx) i), (((generatedRoot.w_260_5 ctx) i), ())))))))
  t285 : (20054016) = 20054016
  h261_5 : (∀ i : Fin 20054016, parallel_generatedRoot_261 backend hashModel tape (path ++ [261, i.val]) params i ((generatedRoot.w_260_0 ctx), ((generatedRoot.w_260_1 ctx), ((generatedRoot.w_260_2 ctx), ((generatedRoot.w_260_3 ctx), ((generatedRoot.w_260_4 ctx), ((generatedRoot.w_260_5 ctx), ())))))) (((generatedRoot.w_261_0 ctx) i), (((generatedRoot.w_261_1 ctx) i), (((generatedRoot.w_261_2 ctx) i), (((generatedRoot.w_261_3 ctx) i), (((generatedRoot.w_261_4 ctx) i), (((generatedRoot.w_261_5 ctx) i), ())))))))
  t287 : (2607022080) = 2607022080

theorem generatedRoot.facts_8  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_8 backend hashModel tape path params (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_229_1 ctx) (generatedRoot.w_229_2 ctx) (generatedRoot.w_229_3 ctx) (generatedRoot.w_229_4 ctx) (generatedRoot.w_229_5 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_232_1 ctx) (generatedRoot.w_232_2 ctx) (generatedRoot.w_232_3 ctx) (generatedRoot.w_232_4 ctx) (generatedRoot.w_232_5 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_233_1 ctx) (generatedRoot.w_233_2 ctx) (generatedRoot.w_233_3 ctx) (generatedRoot.w_233_4 ctx) (generatedRoot.w_233_5 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_236_1 ctx) (generatedRoot.w_236_2 ctx) (generatedRoot.w_236_3 ctx) (generatedRoot.w_236_4 ctx) (generatedRoot.w_236_5 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_237_1 ctx) (generatedRoot.w_237_2 ctx) (generatedRoot.w_237_3 ctx) (generatedRoot.w_237_4 ctx) (generatedRoot.w_237_5 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_240_1 ctx) (generatedRoot.w_240_2 ctx) (generatedRoot.w_240_3 ctx) (generatedRoot.w_240_4 ctx) (generatedRoot.w_240_5 ctx) (generatedRoot.w_241_0 ctx) (generatedRoot.w_241_1 ctx) (generatedRoot.w_241_2 ctx) (generatedRoot.w_241_3 ctx) (generatedRoot.w_241_4 ctx) (generatedRoot.w_241_5 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_244_1 ctx) (generatedRoot.w_244_2 ctx) (generatedRoot.w_244_3 ctx) (generatedRoot.w_244_4 ctx) (generatedRoot.w_244_5 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_245_1 ctx) (generatedRoot.w_245_2 ctx) (generatedRoot.w_245_3 ctx) (generatedRoot.w_245_4 ctx) (generatedRoot.w_245_5 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_248_1 ctx) (generatedRoot.w_248_2 ctx) (generatedRoot.w_248_3 ctx) (generatedRoot.w_248_4 ctx) (generatedRoot.w_248_5 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_249_1 ctx) (generatedRoot.w_249_2 ctx) (generatedRoot.w_249_3 ctx) (generatedRoot.w_249_4 ctx) (generatedRoot.w_249_5 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_252_1 ctx) (generatedRoot.w_252_2 ctx) (generatedRoot.w_252_3 ctx) (generatedRoot.w_252_4 ctx) (generatedRoot.w_252_5 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_253_1 ctx) (generatedRoot.w_253_2 ctx) (generatedRoot.w_253_3 ctx) (generatedRoot.w_253_4 ctx) (generatedRoot.w_253_5 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_256_1 ctx) (generatedRoot.w_256_2 ctx) (generatedRoot.w_256_3 ctx) (generatedRoot.w_256_4 ctx) (generatedRoot.w_256_5 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_257_1 ctx) (generatedRoot.w_257_2 ctx) (generatedRoot.w_257_3 ctx) (generatedRoot.w_257_4 ctx) (generatedRoot.w_257_5 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_260_1 ctx) (generatedRoot.w_260_2 ctx) (generatedRoot.w_260_3 ctx) (generatedRoot.w_260_4 ctx) (generatedRoot.w_260_5 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_261_1 ctx) (generatedRoot.w_261_2 ctx) (generatedRoot.w_261_3 ctx) (generatedRoot.w_261_4 ctx) (generatedRoot.w_261_5 ctx)) :
    generatedRoot.Facts_8 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_9  (ctx : generatedRootCtx) : Prop where
  h264_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_145_0 ctx) i) ∧ ((generatedRoot.w_145_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_145_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_145_0 ctx) i) ∧ ((generatedRoot.w_145_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_145_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_145_0 ctx) i) ∧ ((generatedRoot.w_145_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_145_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_264 backend hashModel tape (path ++ [264, i.val]) params i (((generatedRoot.w_145_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_261_0 ctx), ((generatedRoot.w_261_2 ctx), ((generatedRoot.w_261_4 ctx), (w_43_0_gathered, ((generatedRoot.w_261_1 ctx), ((generatedRoot.w_261_3 ctx), ((generatedRoot.w_261_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_262_0 ctx) i), (((generatedRoot.w_263_0 ctx) i), ())))))))))))))) (((generatedRoot.w_264_0 ctx) i), (((generatedRoot.w_264_1 ctx) i), (((generatedRoot.w_264_2 ctx) i), (((generatedRoot.w_264_3 ctx) i), (((generatedRoot.w_264_4 ctx) i), (((generatedRoot.w_264_5 ctx) i), ())))))))
  t289 : (20054016) = 20054016
  h265_5 : (∀ i : Fin 20054016, parallel_generatedRoot_265 backend hashModel tape (path ++ [265, i.val]) params i ((generatedRoot.w_264_0 ctx), ((generatedRoot.w_264_1 ctx), ((generatedRoot.w_264_2 ctx), ((generatedRoot.w_264_3 ctx), ((generatedRoot.w_264_4 ctx), ((generatedRoot.w_264_5 ctx), ())))))) (((generatedRoot.w_265_0 ctx) i), (((generatedRoot.w_265_1 ctx) i), (((generatedRoot.w_265_2 ctx) i), (((generatedRoot.w_265_3 ctx) i), (((generatedRoot.w_265_4 ctx) i), (((generatedRoot.w_265_5 ctx) i), ())))))))
  t291 : (2607022080) = 2607022080
  h268_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_144_0 ctx) i) ∧ ((generatedRoot.w_144_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_144_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_144_0 ctx) i) ∧ ((generatedRoot.w_144_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_144_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_144_0 ctx) i) ∧ ((generatedRoot.w_144_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_144_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_268 backend hashModel tape (path ++ [268, i.val]) params i (((generatedRoot.w_144_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_265_0 ctx), ((generatedRoot.w_265_2 ctx), ((generatedRoot.w_265_4 ctx), (w_43_0_gathered, ((generatedRoot.w_265_1 ctx), ((generatedRoot.w_265_3 ctx), ((generatedRoot.w_265_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_266_0 ctx) i), (((generatedRoot.w_267_0 ctx) i), ())))))))))))))) (((generatedRoot.w_268_0 ctx) i), (((generatedRoot.w_268_1 ctx) i), (((generatedRoot.w_268_2 ctx) i), (((generatedRoot.w_268_3 ctx) i), (((generatedRoot.w_268_4 ctx) i), (((generatedRoot.w_268_5 ctx) i), ())))))))
  t293 : (20054016) = 20054016
  h269_5 : (∀ i : Fin 20054016, parallel_generatedRoot_269 backend hashModel tape (path ++ [269, i.val]) params i ((generatedRoot.w_268_0 ctx), ((generatedRoot.w_268_1 ctx), ((generatedRoot.w_268_2 ctx), ((generatedRoot.w_268_3 ctx), ((generatedRoot.w_268_4 ctx), ((generatedRoot.w_268_5 ctx), ())))))) (((generatedRoot.w_269_0 ctx) i), (((generatedRoot.w_269_1 ctx) i), (((generatedRoot.w_269_2 ctx) i), (((generatedRoot.w_269_3 ctx) i), (((generatedRoot.w_269_4 ctx) i), (((generatedRoot.w_269_5 ctx) i), ())))))))
  t295 : (2607022080) = 2607022080
  h272_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_143_0 ctx) i) ∧ ((generatedRoot.w_143_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_143_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_143_0 ctx) i) ∧ ((generatedRoot.w_143_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_143_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_143_0 ctx) i) ∧ ((generatedRoot.w_143_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_143_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_272 backend hashModel tape (path ++ [272, i.val]) params i (((generatedRoot.w_143_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_269_0 ctx), ((generatedRoot.w_269_2 ctx), ((generatedRoot.w_269_4 ctx), (w_43_0_gathered, ((generatedRoot.w_269_1 ctx), ((generatedRoot.w_269_3 ctx), ((generatedRoot.w_269_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_270_0 ctx) i), (((generatedRoot.w_271_0 ctx) i), ())))))))))))))) (((generatedRoot.w_272_0 ctx) i), (((generatedRoot.w_272_1 ctx) i), (((generatedRoot.w_272_2 ctx) i), (((generatedRoot.w_272_3 ctx) i), (((generatedRoot.w_272_4 ctx) i), (((generatedRoot.w_272_5 ctx) i), ())))))))
  t297 : (20054016) = 20054016
  h273_5 : (∀ i : Fin 20054016, parallel_generatedRoot_273 backend hashModel tape (path ++ [273, i.val]) params i ((generatedRoot.w_272_0 ctx), ((generatedRoot.w_272_1 ctx), ((generatedRoot.w_272_2 ctx), ((generatedRoot.w_272_3 ctx), ((generatedRoot.w_272_4 ctx), ((generatedRoot.w_272_5 ctx), ())))))) (((generatedRoot.w_273_0 ctx) i), (((generatedRoot.w_273_1 ctx) i), (((generatedRoot.w_273_2 ctx) i), (((generatedRoot.w_273_3 ctx) i), (((generatedRoot.w_273_4 ctx) i), (((generatedRoot.w_273_5 ctx) i), ())))))))
  t299 : (2607022080) = 2607022080
  h276_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_142_0 ctx) i) ∧ ((generatedRoot.w_142_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_142_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_142_0 ctx) i) ∧ ((generatedRoot.w_142_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_142_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_142_0 ctx) i) ∧ ((generatedRoot.w_142_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_142_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_276 backend hashModel tape (path ++ [276, i.val]) params i (((generatedRoot.w_142_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_273_0 ctx), ((generatedRoot.w_273_2 ctx), ((generatedRoot.w_273_4 ctx), (w_43_0_gathered, ((generatedRoot.w_273_1 ctx), ((generatedRoot.w_273_3 ctx), ((generatedRoot.w_273_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_274_0 ctx) i), (((generatedRoot.w_275_0 ctx) i), ())))))))))))))) (((generatedRoot.w_276_0 ctx) i), (((generatedRoot.w_276_1 ctx) i), (((generatedRoot.w_276_2 ctx) i), (((generatedRoot.w_276_3 ctx) i), (((generatedRoot.w_276_4 ctx) i), (((generatedRoot.w_276_5 ctx) i), ())))))))
  t301 : (20054016) = 20054016
  h277_5 : (∀ i : Fin 20054016, parallel_generatedRoot_277 backend hashModel tape (path ++ [277, i.val]) params i ((generatedRoot.w_276_0 ctx), ((generatedRoot.w_276_1 ctx), ((generatedRoot.w_276_2 ctx), ((generatedRoot.w_276_3 ctx), ((generatedRoot.w_276_4 ctx), ((generatedRoot.w_276_5 ctx), ())))))) (((generatedRoot.w_277_0 ctx) i), (((generatedRoot.w_277_1 ctx) i), (((generatedRoot.w_277_2 ctx) i), (((generatedRoot.w_277_3 ctx) i), (((generatedRoot.w_277_4 ctx) i), (((generatedRoot.w_277_5 ctx) i), ())))))))
  t303 : (2607022080) = 2607022080
  h280_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_141_0 ctx) i) ∧ ((generatedRoot.w_141_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_141_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_141_0 ctx) i) ∧ ((generatedRoot.w_141_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_141_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_141_0 ctx) i) ∧ ((generatedRoot.w_141_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_141_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_280 backend hashModel tape (path ++ [280, i.val]) params i (((generatedRoot.w_141_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_277_0 ctx), ((generatedRoot.w_277_2 ctx), ((generatedRoot.w_277_4 ctx), (w_43_0_gathered, ((generatedRoot.w_277_1 ctx), ((generatedRoot.w_277_3 ctx), ((generatedRoot.w_277_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_278_0 ctx) i), (((generatedRoot.w_279_0 ctx) i), ())))))))))))))) (((generatedRoot.w_280_0 ctx) i), (((generatedRoot.w_280_1 ctx) i), (((generatedRoot.w_280_2 ctx) i), (((generatedRoot.w_280_3 ctx) i), (((generatedRoot.w_280_4 ctx) i), (((generatedRoot.w_280_5 ctx) i), ())))))))
  t305 : (20054016) = 20054016
  h281_5 : (∀ i : Fin 20054016, parallel_generatedRoot_281 backend hashModel tape (path ++ [281, i.val]) params i ((generatedRoot.w_280_0 ctx), ((generatedRoot.w_280_1 ctx), ((generatedRoot.w_280_2 ctx), ((generatedRoot.w_280_3 ctx), ((generatedRoot.w_280_4 ctx), ((generatedRoot.w_280_5 ctx), ())))))) (((generatedRoot.w_281_0 ctx) i), (((generatedRoot.w_281_1 ctx) i), (((generatedRoot.w_281_2 ctx) i), (((generatedRoot.w_281_3 ctx) i), (((generatedRoot.w_281_4 ctx) i), (((generatedRoot.w_281_5 ctx) i), ())))))))
  t307 : (2607022080) = 2607022080
  h284_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_140_0 ctx) i) ∧ ((generatedRoot.w_140_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_140_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_140_0 ctx) i) ∧ ((generatedRoot.w_140_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_140_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_140_0 ctx) i) ∧ ((generatedRoot.w_140_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_140_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_284 backend hashModel tape (path ++ [284, i.val]) params i (((generatedRoot.w_140_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_281_0 ctx), ((generatedRoot.w_281_2 ctx), ((generatedRoot.w_281_4 ctx), (w_43_0_gathered, ((generatedRoot.w_281_1 ctx), ((generatedRoot.w_281_3 ctx), ((generatedRoot.w_281_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_282_0 ctx) i), (((generatedRoot.w_283_0 ctx) i), ())))))))))))))) (((generatedRoot.w_284_0 ctx) i), (((generatedRoot.w_284_1 ctx) i), (((generatedRoot.w_284_2 ctx) i), (((generatedRoot.w_284_3 ctx) i), (((generatedRoot.w_284_4 ctx) i), (((generatedRoot.w_284_5 ctx) i), ())))))))
  t309 : (20054016) = 20054016
  h285_5 : (∀ i : Fin 20054016, parallel_generatedRoot_285 backend hashModel tape (path ++ [285, i.val]) params i ((generatedRoot.w_284_0 ctx), ((generatedRoot.w_284_1 ctx), ((generatedRoot.w_284_2 ctx), ((generatedRoot.w_284_3 ctx), ((generatedRoot.w_284_4 ctx), ((generatedRoot.w_284_5 ctx), ())))))) (((generatedRoot.w_285_0 ctx) i), (((generatedRoot.w_285_1 ctx) i), (((generatedRoot.w_285_2 ctx) i), (((generatedRoot.w_285_3 ctx) i), (((generatedRoot.w_285_4 ctx) i), (((generatedRoot.w_285_5 ctx) i), ())))))))
  t311 : (2607022080) = 2607022080
  h288_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_139_0 ctx) i) ∧ ((generatedRoot.w_139_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_139_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_139_0 ctx) i) ∧ ((generatedRoot.w_139_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_139_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_139_0 ctx) i) ∧ ((generatedRoot.w_139_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_139_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_288 backend hashModel tape (path ++ [288, i.val]) params i (((generatedRoot.w_139_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_285_0 ctx), ((generatedRoot.w_285_2 ctx), ((generatedRoot.w_285_4 ctx), (w_43_0_gathered, ((generatedRoot.w_285_1 ctx), ((generatedRoot.w_285_3 ctx), ((generatedRoot.w_285_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_286_0 ctx) i), (((generatedRoot.w_287_0 ctx) i), ())))))))))))))) (((generatedRoot.w_288_0 ctx) i), (((generatedRoot.w_288_1 ctx) i), (((generatedRoot.w_288_2 ctx) i), (((generatedRoot.w_288_3 ctx) i), (((generatedRoot.w_288_4 ctx) i), (((generatedRoot.w_288_5 ctx) i), ())))))))
  t313 : (20054016) = 20054016
  h289_5 : (∀ i : Fin 20054016, parallel_generatedRoot_289 backend hashModel tape (path ++ [289, i.val]) params i ((generatedRoot.w_288_0 ctx), ((generatedRoot.w_288_1 ctx), ((generatedRoot.w_288_2 ctx), ((generatedRoot.w_288_3 ctx), ((generatedRoot.w_288_4 ctx), ((generatedRoot.w_288_5 ctx), ())))))) (((generatedRoot.w_289_0 ctx) i), (((generatedRoot.w_289_1 ctx) i), (((generatedRoot.w_289_2 ctx) i), (((generatedRoot.w_289_3 ctx) i), (((generatedRoot.w_289_4 ctx) i), (((generatedRoot.w_289_5 ctx) i), ())))))))
  t315 : (2607022080) = 2607022080
  h292_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_138_0 ctx) i) ∧ ((generatedRoot.w_138_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_138_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_138_0 ctx) i) ∧ ((generatedRoot.w_138_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_138_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_138_0 ctx) i) ∧ ((generatedRoot.w_138_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_138_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_292 backend hashModel tape (path ++ [292, i.val]) params i (((generatedRoot.w_138_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_289_0 ctx), ((generatedRoot.w_289_2 ctx), ((generatedRoot.w_289_4 ctx), (w_43_0_gathered, ((generatedRoot.w_289_1 ctx), ((generatedRoot.w_289_3 ctx), ((generatedRoot.w_289_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_290_0 ctx) i), (((generatedRoot.w_291_0 ctx) i), ())))))))))))))) (((generatedRoot.w_292_0 ctx) i), (((generatedRoot.w_292_1 ctx) i), (((generatedRoot.w_292_2 ctx) i), (((generatedRoot.w_292_3 ctx) i), (((generatedRoot.w_292_4 ctx) i), (((generatedRoot.w_292_5 ctx) i), ())))))))
  t317 : (20054016) = 20054016
  h293_5 : (∀ i : Fin 20054016, parallel_generatedRoot_293 backend hashModel tape (path ++ [293, i.val]) params i ((generatedRoot.w_292_0 ctx), ((generatedRoot.w_292_1 ctx), ((generatedRoot.w_292_2 ctx), ((generatedRoot.w_292_3 ctx), ((generatedRoot.w_292_4 ctx), ((generatedRoot.w_292_5 ctx), ())))))) (((generatedRoot.w_293_0 ctx) i), (((generatedRoot.w_293_1 ctx) i), (((generatedRoot.w_293_2 ctx) i), (((generatedRoot.w_293_3 ctx) i), (((generatedRoot.w_293_4 ctx) i), (((generatedRoot.w_293_5 ctx) i), ())))))))
  t319 : (2607022080) = 2607022080

theorem generatedRoot.facts_9  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_9 backend hashModel tape path params (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_266_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_270_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_278_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_286_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_261_1 ctx) (generatedRoot.w_261_2 ctx) (generatedRoot.w_261_3 ctx) (generatedRoot.w_261_4 ctx) (generatedRoot.w_261_5 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_264_1 ctx) (generatedRoot.w_264_2 ctx) (generatedRoot.w_264_3 ctx) (generatedRoot.w_264_4 ctx) (generatedRoot.w_264_5 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_265_1 ctx) (generatedRoot.w_265_2 ctx) (generatedRoot.w_265_3 ctx) (generatedRoot.w_265_4 ctx) (generatedRoot.w_265_5 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_268_1 ctx) (generatedRoot.w_268_2 ctx) (generatedRoot.w_268_3 ctx) (generatedRoot.w_268_4 ctx) (generatedRoot.w_268_5 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_269_1 ctx) (generatedRoot.w_269_2 ctx) (generatedRoot.w_269_3 ctx) (generatedRoot.w_269_4 ctx) (generatedRoot.w_269_5 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_272_1 ctx) (generatedRoot.w_272_2 ctx) (generatedRoot.w_272_3 ctx) (generatedRoot.w_272_4 ctx) (generatedRoot.w_272_5 ctx) (generatedRoot.w_273_0 ctx) (generatedRoot.w_273_1 ctx) (generatedRoot.w_273_2 ctx) (generatedRoot.w_273_3 ctx) (generatedRoot.w_273_4 ctx) (generatedRoot.w_273_5 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_276_1 ctx) (generatedRoot.w_276_2 ctx) (generatedRoot.w_276_3 ctx) (generatedRoot.w_276_4 ctx) (generatedRoot.w_276_5 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_277_1 ctx) (generatedRoot.w_277_2 ctx) (generatedRoot.w_277_3 ctx) (generatedRoot.w_277_4 ctx) (generatedRoot.w_277_5 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_280_1 ctx) (generatedRoot.w_280_2 ctx) (generatedRoot.w_280_3 ctx) (generatedRoot.w_280_4 ctx) (generatedRoot.w_280_5 ctx) (generatedRoot.w_281_0 ctx) (generatedRoot.w_281_1 ctx) (generatedRoot.w_281_2 ctx) (generatedRoot.w_281_3 ctx) (generatedRoot.w_281_4 ctx) (generatedRoot.w_281_5 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_284_1 ctx) (generatedRoot.w_284_2 ctx) (generatedRoot.w_284_3 ctx) (generatedRoot.w_284_4 ctx) (generatedRoot.w_284_5 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_285_1 ctx) (generatedRoot.w_285_2 ctx) (generatedRoot.w_285_3 ctx) (generatedRoot.w_285_4 ctx) (generatedRoot.w_285_5 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_288_1 ctx) (generatedRoot.w_288_2 ctx) (generatedRoot.w_288_3 ctx) (generatedRoot.w_288_4 ctx) (generatedRoot.w_288_5 ctx) (generatedRoot.w_289_0 ctx) (generatedRoot.w_289_1 ctx) (generatedRoot.w_289_2 ctx) (generatedRoot.w_289_3 ctx) (generatedRoot.w_289_4 ctx) (generatedRoot.w_289_5 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_292_1 ctx) (generatedRoot.w_292_2 ctx) (generatedRoot.w_292_3 ctx) (generatedRoot.w_292_4 ctx) (generatedRoot.w_292_5 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_293_1 ctx) (generatedRoot.w_293_2 ctx) (generatedRoot.w_293_3 ctx) (generatedRoot.w_293_4 ctx) (generatedRoot.w_293_5 ctx)) :
    generatedRoot.Facts_9 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_10  (ctx : generatedRootCtx) : Prop where
  h296_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_137_0 ctx) i) ∧ ((generatedRoot.w_137_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_137_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_137_0 ctx) i) ∧ ((generatedRoot.w_137_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_137_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_137_0 ctx) i) ∧ ((generatedRoot.w_137_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_137_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_296 backend hashModel tape (path ++ [296, i.val]) params i (((generatedRoot.w_137_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_293_0 ctx), ((generatedRoot.w_293_2 ctx), ((generatedRoot.w_293_4 ctx), (w_43_0_gathered, ((generatedRoot.w_293_1 ctx), ((generatedRoot.w_293_3 ctx), ((generatedRoot.w_293_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_294_0 ctx) i), (((generatedRoot.w_295_0 ctx) i), ())))))))))))))) (((generatedRoot.w_296_0 ctx) i), (((generatedRoot.w_296_1 ctx) i), (((generatedRoot.w_296_2 ctx) i), (((generatedRoot.w_296_3 ctx) i), (((generatedRoot.w_296_4 ctx) i), (((generatedRoot.w_296_5 ctx) i), ())))))))
  t321 : (20054016) = 20054016
  h297_5 : (∀ i : Fin 20054016, parallel_generatedRoot_297 backend hashModel tape (path ++ [297, i.val]) params i ((generatedRoot.w_296_0 ctx), ((generatedRoot.w_296_1 ctx), ((generatedRoot.w_296_2 ctx), ((generatedRoot.w_296_3 ctx), ((generatedRoot.w_296_4 ctx), ((generatedRoot.w_296_5 ctx), ())))))) (((generatedRoot.w_297_0 ctx) i), (((generatedRoot.w_297_1 ctx) i), (((generatedRoot.w_297_2 ctx) i), (((generatedRoot.w_297_3 ctx) i), (((generatedRoot.w_297_4 ctx) i), (((generatedRoot.w_297_5 ctx) i), ())))))))
  t323 : (2607022080) = 2607022080
  h300_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_136_0 ctx) i) ∧ ((generatedRoot.w_136_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_136_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_136_0 ctx) i) ∧ ((generatedRoot.w_136_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_136_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_136_0 ctx) i) ∧ ((generatedRoot.w_136_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_136_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_300 backend hashModel tape (path ++ [300, i.val]) params i (((generatedRoot.w_136_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_297_0 ctx), ((generatedRoot.w_297_2 ctx), ((generatedRoot.w_297_4 ctx), (w_43_0_gathered, ((generatedRoot.w_297_1 ctx), ((generatedRoot.w_297_3 ctx), ((generatedRoot.w_297_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_298_0 ctx) i), (((generatedRoot.w_299_0 ctx) i), ())))))))))))))) (((generatedRoot.w_300_0 ctx) i), (((generatedRoot.w_300_1 ctx) i), (((generatedRoot.w_300_2 ctx) i), (((generatedRoot.w_300_3 ctx) i), (((generatedRoot.w_300_4 ctx) i), (((generatedRoot.w_300_5 ctx) i), ())))))))
  t325 : (20054016) = 20054016
  h301_5 : (∀ i : Fin 20054016, parallel_generatedRoot_301 backend hashModel tape (path ++ [301, i.val]) params i ((generatedRoot.w_300_0 ctx), ((generatedRoot.w_300_1 ctx), ((generatedRoot.w_300_2 ctx), ((generatedRoot.w_300_3 ctx), ((generatedRoot.w_300_4 ctx), ((generatedRoot.w_300_5 ctx), ())))))) (((generatedRoot.w_301_0 ctx) i), (((generatedRoot.w_301_1 ctx) i), (((generatedRoot.w_301_2 ctx) i), (((generatedRoot.w_301_3 ctx) i), (((generatedRoot.w_301_4 ctx) i), (((generatedRoot.w_301_5 ctx) i), ())))))))
  t327 : (2607022080) = 2607022080
  h304_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_135_0 ctx) i) ∧ ((generatedRoot.w_135_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_135_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_135_0 ctx) i) ∧ ((generatedRoot.w_135_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_135_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_135_0 ctx) i) ∧ ((generatedRoot.w_135_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_135_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_304 backend hashModel tape (path ++ [304, i.val]) params i (((generatedRoot.w_135_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_301_0 ctx), ((generatedRoot.w_301_2 ctx), ((generatedRoot.w_301_4 ctx), (w_43_0_gathered, ((generatedRoot.w_301_1 ctx), ((generatedRoot.w_301_3 ctx), ((generatedRoot.w_301_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_302_0 ctx) i), (((generatedRoot.w_303_0 ctx) i), ())))))))))))))) (((generatedRoot.w_304_0 ctx) i), (((generatedRoot.w_304_1 ctx) i), (((generatedRoot.w_304_2 ctx) i), (((generatedRoot.w_304_3 ctx) i), (((generatedRoot.w_304_4 ctx) i), (((generatedRoot.w_304_5 ctx) i), ())))))))
  t329 : (20054016) = 20054016
  h305_5 : (∀ i : Fin 20054016, parallel_generatedRoot_305 backend hashModel tape (path ++ [305, i.val]) params i ((generatedRoot.w_304_0 ctx), ((generatedRoot.w_304_1 ctx), ((generatedRoot.w_304_2 ctx), ((generatedRoot.w_304_3 ctx), ((generatedRoot.w_304_4 ctx), ((generatedRoot.w_304_5 ctx), ())))))) (((generatedRoot.w_305_0 ctx) i), (((generatedRoot.w_305_1 ctx) i), (((generatedRoot.w_305_2 ctx) i), (((generatedRoot.w_305_3 ctx) i), (((generatedRoot.w_305_4 ctx) i), (((generatedRoot.w_305_5 ctx) i), ())))))))
  t331 : (2607022080) = 2607022080
  h308_5 : (∀ i : Fin 2607022080, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat7, 0 ≤ ((generatedRoot.w_134_0 ctx) i) ∧ ((generatedRoot.w_134_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_12_0 ctx) ((generatedRoot.w_134_0 ctx) i) w_12_0_gathered ∧ 0 ≤ ((generatedRoot.w_134_0 ctx) i) ∧ ((generatedRoot.w_134_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_23_0 ctx) ((generatedRoot.w_134_0 ctx) i) w_23_0_gathered ∧ 0 ≤ ((generatedRoot.w_134_0 ctx) i) ∧ ((generatedRoot.w_134_0 ctx) i) < 1025 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_43_0 ctx) ((generatedRoot.w_134_0 ctx) i) w_43_0_gathered ∧ parallel_generatedRoot_308 backend hashModel tape (path ++ [308, i.val]) params i (((generatedRoot.w_134_0 ctx) i), (w_12_0_gathered, ((generatedRoot.w_10_0 ctx), (w_23_0_gathered, ((generatedRoot.w_305_0 ctx), ((generatedRoot.w_305_2 ctx), ((generatedRoot.w_305_4 ctx), (w_43_0_gathered, ((generatedRoot.w_305_1 ctx), ((generatedRoot.w_305_3 ctx), ((generatedRoot.w_305_5 ctx), ((generatedRoot.w_66_0 ctx), (((generatedRoot.w_306_0 ctx) i), (((generatedRoot.w_307_0 ctx) i), ())))))))))))))) (((generatedRoot.w_308_0 ctx) i), (((generatedRoot.w_308_1 ctx) i), (((generatedRoot.w_308_2 ctx) i), (((generatedRoot.w_308_3 ctx) i), (((generatedRoot.w_308_4 ctx) i), (((generatedRoot.w_308_5 ctx) i), ())))))))
  t333 : (20054016) = 20054016
  h309_5 : (∀ i : Fin 20054016, parallel_generatedRoot_309 backend hashModel tape (path ++ [309, i.val]) params i ((generatedRoot.w_308_0 ctx), ((generatedRoot.w_308_1 ctx), ((generatedRoot.w_308_2 ctx), ((generatedRoot.w_308_3 ctx), ((generatedRoot.w_308_4 ctx), ((generatedRoot.w_308_5 ctx), ())))))) (((generatedRoot.w_309_0 ctx) i), (((generatedRoot.w_309_1 ctx) i), (((generatedRoot.w_309_2 ctx) i), (((generatedRoot.w_309_3 ctx) i), (((generatedRoot.w_309_4 ctx) i), (((generatedRoot.w_309_5 ctx) i), ())))))))
  t335 : (20054016) = 20054016
  h310_5 : (∀ i : Fin 20054016, parallel_generatedRoot_310 backend hashModel tape (path ++ [310, i.val]) params i ((generatedRoot.w_128_0 ctx), ((generatedRoot.w_129_0 ctx), ((generatedRoot.w_130_0 ctx), ((generatedRoot.w_131_0 ctx), ((generatedRoot.w_132_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_133_0 ctx), (((generatedRoot.w_309_4 ctx) i), (((generatedRoot.w_309_0 ctx) i), (((generatedRoot.w_309_2 ctx) i), (((generatedRoot.w_309_5 ctx) i), (((generatedRoot.w_309_1 ctx) i), (((generatedRoot.w_309_3 ctx) i), ()))))))))))))) (((generatedRoot.w_310_0 ctx) i), (((generatedRoot.w_310_1 ctx) i), (((generatedRoot.w_310_2 ctx) i), (((generatedRoot.w_310_3 ctx) i), (((generatedRoot.w_310_4 ctx) i), (((generatedRoot.w_310_5 ctx) i), ())))))))
  t337 : (20054016) = 20054016
  h311_5 : (∀ i : Fin 20054016, parallel_generatedRoot_311 backend hashModel tape (path ++ [311, i.val]) params i ((generatedRoot.w_310_0 ctx), ((generatedRoot.w_310_1 ctx), ((generatedRoot.w_310_2 ctx), ((generatedRoot.w_310_3 ctx), ((generatedRoot.w_310_4 ctx), ((generatedRoot.w_310_5 ctx), ())))))) (((generatedRoot.w_311_0 ctx) i), (((generatedRoot.w_311_1 ctx) i), (((generatedRoot.w_311_2 ctx) i), (((generatedRoot.w_311_3 ctx) i), (((generatedRoot.w_311_4 ctx) i), (((generatedRoot.w_311_5 ctx) i), ())))))))
  h312_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_0 ctx) (0) (generatedRoot.w_312_0 ctx)
  h313_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_0 ctx) (1) (generatedRoot.w_313_0 ctx)
  h315_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_1 ctx) (0) (generatedRoot.w_315_0 ctx)
  h316_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_1 ctx) (1) (generatedRoot.w_316_0 ctx)
  h318_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_2 ctx) (0) (generatedRoot.w_318_0 ctx)
  h319_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_2 ctx) (1) (generatedRoot.w_319_0 ctx)
  h321_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_3 ctx) (0) (generatedRoot.w_321_0 ctx)
  h322_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_3 ctx) (1) (generatedRoot.w_322_0 ctx)
  h324_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_4 ctx) (0) (generatedRoot.w_324_0 ctx)
  h325_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_4 ctx) (1) (generatedRoot.w_325_0 ctx)
  h327_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_5 ctx) (0) (generatedRoot.w_327_0 ctx)
  h328_0 : MxxRuntime.familyGetStatic (generatedRoot.w_311_5 ctx) (1) (generatedRoot.w_328_0 ctx)
  t351 : 0 ≤ (65534)

theorem generatedRoot.facts_10  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_10 backend hashModel tape path params (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_293_1 ctx) (generatedRoot.w_293_2 ctx) (generatedRoot.w_293_3 ctx) (generatedRoot.w_293_4 ctx) (generatedRoot.w_293_5 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_296_1 ctx) (generatedRoot.w_296_2 ctx) (generatedRoot.w_296_3 ctx) (generatedRoot.w_296_4 ctx) (generatedRoot.w_296_5 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_297_1 ctx) (generatedRoot.w_297_2 ctx) (generatedRoot.w_297_3 ctx) (generatedRoot.w_297_4 ctx) (generatedRoot.w_297_5 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_300_1 ctx) (generatedRoot.w_300_2 ctx) (generatedRoot.w_300_3 ctx) (generatedRoot.w_300_4 ctx) (generatedRoot.w_300_5 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_301_1 ctx) (generatedRoot.w_301_2 ctx) (generatedRoot.w_301_3 ctx) (generatedRoot.w_301_4 ctx) (generatedRoot.w_301_5 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_304_1 ctx) (generatedRoot.w_304_2 ctx) (generatedRoot.w_304_3 ctx) (generatedRoot.w_304_4 ctx) (generatedRoot.w_304_5 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_305_1 ctx) (generatedRoot.w_305_2 ctx) (generatedRoot.w_305_3 ctx) (generatedRoot.w_305_4 ctx) (generatedRoot.w_305_5 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_308_1 ctx) (generatedRoot.w_308_2 ctx) (generatedRoot.w_308_3 ctx) (generatedRoot.w_308_4 ctx) (generatedRoot.w_308_5 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_309_1 ctx) (generatedRoot.w_309_2 ctx) (generatedRoot.w_309_3 ctx) (generatedRoot.w_309_4 ctx) (generatedRoot.w_309_5 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_310_1 ctx) (generatedRoot.w_310_2 ctx) (generatedRoot.w_310_3 ctx) (generatedRoot.w_310_4 ctx) (generatedRoot.w_310_5 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_311_1 ctx) (generatedRoot.w_311_2 ctx) (generatedRoot.w_311_3 ctx) (generatedRoot.w_311_4 ctx) (generatedRoot.w_311_5 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_325_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx)) :
    generatedRoot.Facts_10 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_11  (ctx : generatedRootCtx) : Prop where
  h330_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_330 backend hashModel tape (path ++ [330, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_311_0 ctx), ((generatedRoot.w_311_1 ctx), ((generatedRoot.w_311_2 ctx), ((generatedRoot.w_311_3 ctx), ((generatedRoot.w_311_4 ctx), ((generatedRoot.w_311_5 ctx), ())))))))))))) next) (Int.toNat (65534)) ((generatedRoot.w_314_0 ctx), ((generatedRoot.w_317_0 ctx), ((generatedRoot.w_320_0 ctx), ((generatedRoot.w_323_0 ctx), ((generatedRoot.w_326_0 ctx), ((generatedRoot.w_329_0 ctx), ())))))) ((generatedRoot.w_330_0 ctx), ((generatedRoot.w_330_1 ctx), ((generatedRoot.w_330_2 ctx), ((generatedRoot.w_330_3 ctx), ((generatedRoot.w_330_4 ctx), ((generatedRoot.w_330_5 ctx), ()))))))
  h331_0 : MxxRuntime.tensorRuns (generatedRoot.w_127_0 ctx) (generatedRoot.w_330_0 ctx) (generatedRoot.w_331_0 ctx)
  t354 : 0 ≤ (0)
  t355 : (0) < 306
  h333_0 : MxxRuntime.tensorRuns (generatedRoot.w_332_0 ctx) (generatedRoot.w_330_1 ctx) (generatedRoot.w_333_0 ctx)
  t357 : 0 ≤ (0)
  t358 : (0) < 306
  h335_0 : MxxRuntime.tensorRuns (generatedRoot.w_334_0 ctx) (generatedRoot.w_330_2 ctx) (generatedRoot.w_335_0 ctx)
  t360 : 0 ≤ (0)
  t361 : (0) < 306
  h337_0 : MxxRuntime.tensorRuns (generatedRoot.w_336_0 ctx) (generatedRoot.w_330_3 ctx) (generatedRoot.w_337_0 ctx)
  t363 : 0 ≤ (0)
  t364 : (0) < 306
  h339_0 : MxxRuntime.tensorRuns (generatedRoot.w_338_0 ctx) (generatedRoot.w_330_4 ctx) (generatedRoot.w_339_0 ctx)
  t366 : 0 ≤ (0)
  t367 : (0) < 306
  h341_0 : MxxRuntime.tensorRuns (generatedRoot.w_340_0 ctx) (generatedRoot.w_330_5 ctx) (generatedRoot.w_341_0 ctx)
  t369 : 0 ≤ (305)
  h342_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × mxxMat5 × Unit) => sequential_generatedRoot_342 backend hashModel tape (path ++ [342, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_311_0 ctx), ((generatedRoot.w_311_1 ctx), ((generatedRoot.w_311_2 ctx), ((generatedRoot.w_311_3 ctx), ((generatedRoot.w_311_4 ctx), ((generatedRoot.w_311_5 ctx), ())))))))))))) next) (Int.toNat (305)) ((generatedRoot.w_331_0 ctx), ((generatedRoot.w_333_0 ctx), ((generatedRoot.w_335_0 ctx), ((generatedRoot.w_337_0 ctx), ((generatedRoot.w_339_0 ctx), ((generatedRoot.w_341_0 ctx), ())))))) ((generatedRoot.w_342_0 ctx), ((generatedRoot.w_342_1 ctx), ((generatedRoot.w_342_2 ctx), ((generatedRoot.w_342_3 ctx), ((generatedRoot.w_342_4 ctx), ((generatedRoot.w_342_5 ctx), ()))))))
  t371 : (306) = 306
  h344_0 : (∀ i : Fin 306, parallel_generatedRoot_344 backend hashModel tape (path ++ [344, i.val]) params i (generatedRoot.w_343_0 ctx) ((generatedRoot.w_344_0 ctx) i))
  t373 : 0 ≤ (306)
  h345_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat7 × mxxMat7 × mxxMat7 × Unit) => sequential_generatedRoot_345 backend hashModel tape (path ++ [345, i]) params i (current.1, (current.2.1, (current.2.2.1, ((generatedRoot.w_342_5 ctx), ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_342_1 ctx), ((generatedRoot.w_342_3 ctx), ()))))))) next) (Int.toNat (306)) ((generatedRoot.w_126_0 ctx), ((generatedRoot.w_126_0 ctx), ((generatedRoot.w_126_0 ctx), ()))) ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_345_1 ctx), ((generatedRoot.w_345_2 ctx), ())))
  t375 : 0 ≤ 2
  t376 : 2 < 3
  t377 : 3 ≤ 3
  t378 : 0 ≤ 0
  t379 : 0 < 270
  t380 : 270 ≤ 270
  h347_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 2 3 0 270 (generatedRoot.w_347_0 ctx)
  h351_0 : MxxRuntime.concatRows (generatedRoot.w_346_0 ctx) (generatedRoot.w_350_0 ctx) (generatedRoot.w_351_0 ctx)
  h352_0 : MxxRuntime.concatRows (generatedRoot.w_87_6 ctx) (generatedRoot.w_87_8 ctx) (generatedRoot.w_352_0 ctx)

theorem generatedRoot.facts_11  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_11 backend hashModel tape path params (generatedRoot.w_87_6 ctx) (generatedRoot.w_87_8 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_311_1 ctx) (generatedRoot.w_311_2 ctx) (generatedRoot.w_311_3 ctx) (generatedRoot.w_311_4 ctx) (generatedRoot.w_311_5 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_317_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_326_0 ctx) (generatedRoot.w_329_0 ctx) (generatedRoot.w_330_0 ctx) (generatedRoot.w_330_1 ctx) (generatedRoot.w_330_2 ctx) (generatedRoot.w_330_3 ctx) (generatedRoot.w_330_4 ctx) (generatedRoot.w_330_5 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_333_0 ctx) (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_337_0 ctx) (generatedRoot.w_338_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_341_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_342_1 ctx) (generatedRoot.w_342_2 ctx) (generatedRoot.w_342_3 ctx) (generatedRoot.w_342_4 ctx) (generatedRoot.w_342_5 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_345_1 ctx) (generatedRoot.w_345_2 ctx) (generatedRoot.w_346_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_350_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_352_0 ctx)) :
    generatedRoot.Facts_11 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_12  (ctx : generatedRootCtx) : Prop where
  h353_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_352_0 ctx) (generatedRoot.w_353_0 ctx)
  h354_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_353_0 ctx) (generatedRoot.w_354_0 ctx)
  t386 : 0 ≤ 1
  t387 : 1 < 2
  t388 : 2 ≤ 3
  t389 : 0 ≤ 0
  t390 : 0 < 270
  t391 : 270 ≤ 270
  h356_0 : MxxRuntime.sliceMatrix (generatedRoot.w_355_0 ctx) 1 2 0 270 (generatedRoot.w_356_0 ctx)
  t393 : 0 ≤ 2
  t394 : 2 < 3
  t395 : 3 ≤ 3
  t396 : 0 ≤ 0
  t397 : 0 < 270
  t398 : 270 ≤ 270
  h357_0 : MxxRuntime.sliceMatrix (generatedRoot.w_355_0 ctx) 2 3 0 270 (generatedRoot.w_357_0 ctx)
  h358_0 : MxxRuntime.concatRows (generatedRoot.w_356_0 ctx) (generatedRoot.w_357_0 ctx) (generatedRoot.w_358_0 ctx)
  h359_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx)
  h360_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx)
  t403 : 0 ≤ 1
  t404 : 1 < 2
  t405 : 2 ≤ 3
  t406 : 0 ≤ 0
  t407 : 0 < 270
  t408 : 270 ≤ 270
  h362_0 : MxxRuntime.sliceMatrix (generatedRoot.w_361_0 ctx) 1 2 0 270 (generatedRoot.w_362_0 ctx)
  t410 : 0 ≤ 2
  t411 : 2 < 3
  t412 : 3 ≤ 3
  t413 : 0 ≤ 0
  t414 : 0 < 270
  t415 : 270 ≤ 270

theorem generatedRoot.facts_12  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_12 backend (generatedRoot.w_352_0 ctx) (generatedRoot.w_353_0 ctx) (generatedRoot.w_354_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_357_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_361_0 ctx) (generatedRoot.w_362_0 ctx)) :
    generatedRoot.Facts_12 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_13  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  h364_0 : MxxRuntime.sliceMatrix (generatedRoot.w_361_0 ctx) 2 3 0 270 (generatedRoot.w_364_0 ctx)
  t417 : 0 ≤ 0
  t418 : 0 < 1
  t419 : 1 ≤ 3
  t420 : 0 ≤ 0
  t421 : 0 < 270
  t422 : 270 ≤ 270
  h366_0 : MxxRuntime.sliceMatrix (generatedRoot.w_361_0 ctx) 0 1 0 270 (generatedRoot.w_366_0 ctx)
  t424 : 0 ≤ 0
  t425 : 0 < 1
  t426 : 1 ≤ 3
  t427 : 0 ≤ 0
  t428 : 0 < 270
  t429 : 270 ≤ 270
  h367_0 : MxxRuntime.sliceMatrix (generatedRoot.w_355_0 ctx) 0 1 0 270 (generatedRoot.w_367_0 ctx)
  t431 : 0 ≤ 0
  t432 : 0 < 1
  t433 : 1 ≤ 3
  t434 : 0 ≤ 0
  t435 : 0 < 270
  t436 : 270 ≤ 270
  h370_0 : MxxRuntime.sliceMatrix (generatedRoot.w_116_0 ctx) 0 1 0 270 (generatedRoot.w_370_0 ctx)
  t438 : 0 ≤ 0
  t439 : 0 < 1
  t440 : 1 ≤ 3
  t441 : 0 ≤ 0
  t442 : 0 < 270
  t443 : 270 ≤ 270
  h371_0 : MxxRuntime.sliceMatrix (generatedRoot.w_110_0 ctx) 0 1 0 270 (generatedRoot.w_371_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_13  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_13 (generatedRoot.w_110_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_342_1 ctx) (generatedRoot.w_342_2 ctx) (generatedRoot.w_342_3 ctx) (generatedRoot.w_342_4 ctx) (generatedRoot.w_342_5 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_361_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_365_0 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_369_0 ctx) (generatedRoot.w_370_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_373_0 ctx) outputs) :
    generatedRoot.Facts_13 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

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
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  ⟨(generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h.1.1.1), (generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) h.1.1.2.1), (generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) h.1.1.2.2), (generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) h.1.2.1.1), (generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) h.1.2.1.2), (generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) h.1.2.2.1), (generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) h.1.2.2.2), (generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) h.2.1.1), (generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) h.2.1.2.1), (generatedRoot.facts_9 (ctx := ctx) (outputs := outputs) h.2.1.2.2), (generatedRoot.facts_10 (ctx := ctx) (outputs := outputs) h.2.2.1.1), (generatedRoot.facts_11 (ctx := ctx) (outputs := outputs) h.2.2.1.2), (generatedRoot.facts_12 (ctx := ctx) (outputs := outputs) h.2.2.2.1), (generatedRoot.facts_13 (ctx := ctx) (outputs := outputs) h.2.2.2.2)⟩

theorem generatedRoot.h177_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_177.Witness, parallel_generatedRoot_177.Facts (⟨((generatedRoot.w_176_0 ctx), ((generatedRoot.w_176_1 ctx), ((generatedRoot.w_176_2 ctx), ((generatedRoot.w_176_3 ctx), ((generatedRoot.w_176_4 ctx), ((generatedRoot.w_176_5 ctx), ())))))), w⟩ : parallel_generatedRoot_177.Ctx backend hashModel tape (path ++ [177, i.val]) params i) (((generatedRoot.w_177_0 ctx) i), (((generatedRoot.w_177_1 ctx) i), (((generatedRoot.w_177_2 ctx) i), (((generatedRoot.w_177_3 ctx) i), (((generatedRoot.w_177_4 ctx) i), (((generatedRoot.w_177_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h177_5 i
  ⟨w, parallel_generatedRoot_177.facts _ h⟩

theorem generatedRoot.h181_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_181.Witness, parallel_generatedRoot_181.Facts (⟨((generatedRoot.w_180_0 ctx), ((generatedRoot.w_180_1 ctx), ((generatedRoot.w_180_2 ctx), ((generatedRoot.w_180_3 ctx), ((generatedRoot.w_180_4 ctx), ((generatedRoot.w_180_5 ctx), ())))))), w⟩ : parallel_generatedRoot_181.Ctx backend hashModel tape (path ++ [181, i.val]) params i) (((generatedRoot.w_181_0 ctx) i), (((generatedRoot.w_181_1 ctx) i), (((generatedRoot.w_181_2 ctx) i), (((generatedRoot.w_181_3 ctx) i), (((generatedRoot.w_181_4 ctx) i), (((generatedRoot.w_181_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h181_5 i
  ⟨w, parallel_generatedRoot_181.facts _ h⟩

theorem generatedRoot.h185_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_185.Witness, parallel_generatedRoot_185.Facts (⟨((generatedRoot.w_184_0 ctx), ((generatedRoot.w_184_1 ctx), ((generatedRoot.w_184_2 ctx), ((generatedRoot.w_184_3 ctx), ((generatedRoot.w_184_4 ctx), ((generatedRoot.w_184_5 ctx), ())))))), w⟩ : parallel_generatedRoot_185.Ctx backend hashModel tape (path ++ [185, i.val]) params i) (((generatedRoot.w_185_0 ctx) i), (((generatedRoot.w_185_1 ctx) i), (((generatedRoot.w_185_2 ctx) i), (((generatedRoot.w_185_3 ctx) i), (((generatedRoot.w_185_4 ctx) i), (((generatedRoot.w_185_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h185_5 i
  ⟨w, parallel_generatedRoot_185.facts _ h⟩

theorem generatedRoot.h189_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_189.Witness, parallel_generatedRoot_189.Facts (⟨((generatedRoot.w_188_0 ctx), ((generatedRoot.w_188_1 ctx), ((generatedRoot.w_188_2 ctx), ((generatedRoot.w_188_3 ctx), ((generatedRoot.w_188_4 ctx), ((generatedRoot.w_188_5 ctx), ())))))), w⟩ : parallel_generatedRoot_189.Ctx backend hashModel tape (path ++ [189, i.val]) params i) (((generatedRoot.w_189_0 ctx) i), (((generatedRoot.w_189_1 ctx) i), (((generatedRoot.w_189_2 ctx) i), (((generatedRoot.w_189_3 ctx) i), (((generatedRoot.w_189_4 ctx) i), (((generatedRoot.w_189_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h189_5 i
  ⟨w, parallel_generatedRoot_189.facts _ h⟩

theorem generatedRoot.h193_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_193.Witness, parallel_generatedRoot_193.Facts (⟨((generatedRoot.w_192_0 ctx), ((generatedRoot.w_192_1 ctx), ((generatedRoot.w_192_2 ctx), ((generatedRoot.w_192_3 ctx), ((generatedRoot.w_192_4 ctx), ((generatedRoot.w_192_5 ctx), ())))))), w⟩ : parallel_generatedRoot_193.Ctx backend hashModel tape (path ++ [193, i.val]) params i) (((generatedRoot.w_193_0 ctx) i), (((generatedRoot.w_193_1 ctx) i), (((generatedRoot.w_193_2 ctx) i), (((generatedRoot.w_193_3 ctx) i), (((generatedRoot.w_193_4 ctx) i), (((generatedRoot.w_193_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h193_5 i
  ⟨w, parallel_generatedRoot_193.facts _ h⟩

theorem generatedRoot.h197_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_197.Witness, parallel_generatedRoot_197.Facts (⟨((generatedRoot.w_196_0 ctx), ((generatedRoot.w_196_1 ctx), ((generatedRoot.w_196_2 ctx), ((generatedRoot.w_196_3 ctx), ((generatedRoot.w_196_4 ctx), ((generatedRoot.w_196_5 ctx), ())))))), w⟩ : parallel_generatedRoot_197.Ctx backend hashModel tape (path ++ [197, i.val]) params i) (((generatedRoot.w_197_0 ctx) i), (((generatedRoot.w_197_1 ctx) i), (((generatedRoot.w_197_2 ctx) i), (((generatedRoot.w_197_3 ctx) i), (((generatedRoot.w_197_4 ctx) i), (((generatedRoot.w_197_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c6.h197_5 i
  ⟨w, parallel_generatedRoot_197.facts _ h⟩

theorem generatedRoot.h201_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_201.Witness, parallel_generatedRoot_201.Facts (⟨((generatedRoot.w_200_0 ctx), ((generatedRoot.w_200_1 ctx), ((generatedRoot.w_200_2 ctx), ((generatedRoot.w_200_3 ctx), ((generatedRoot.w_200_4 ctx), ((generatedRoot.w_200_5 ctx), ())))))), w⟩ : parallel_generatedRoot_201.Ctx backend hashModel tape (path ++ [201, i.val]) params i) (((generatedRoot.w_201_0 ctx) i), (((generatedRoot.w_201_1 ctx) i), (((generatedRoot.w_201_2 ctx) i), (((generatedRoot.w_201_3 ctx) i), (((generatedRoot.w_201_4 ctx) i), (((generatedRoot.w_201_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h201_5 i
  ⟨w, parallel_generatedRoot_201.facts _ h⟩

theorem generatedRoot.h205_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_205.Witness, parallel_generatedRoot_205.Facts (⟨((generatedRoot.w_204_0 ctx), ((generatedRoot.w_204_1 ctx), ((generatedRoot.w_204_2 ctx), ((generatedRoot.w_204_3 ctx), ((generatedRoot.w_204_4 ctx), ((generatedRoot.w_204_5 ctx), ())))))), w⟩ : parallel_generatedRoot_205.Ctx backend hashModel tape (path ++ [205, i.val]) params i) (((generatedRoot.w_205_0 ctx) i), (((generatedRoot.w_205_1 ctx) i), (((generatedRoot.w_205_2 ctx) i), (((generatedRoot.w_205_3 ctx) i), (((generatedRoot.w_205_4 ctx) i), (((generatedRoot.w_205_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h205_5 i
  ⟨w, parallel_generatedRoot_205.facts _ h⟩

theorem generatedRoot.h209_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_209.Witness, parallel_generatedRoot_209.Facts (⟨((generatedRoot.w_208_0 ctx), ((generatedRoot.w_208_1 ctx), ((generatedRoot.w_208_2 ctx), ((generatedRoot.w_208_3 ctx), ((generatedRoot.w_208_4 ctx), ((generatedRoot.w_208_5 ctx), ())))))), w⟩ : parallel_generatedRoot_209.Ctx backend hashModel tape (path ++ [209, i.val]) params i) (((generatedRoot.w_209_0 ctx) i), (((generatedRoot.w_209_1 ctx) i), (((generatedRoot.w_209_2 ctx) i), (((generatedRoot.w_209_3 ctx) i), (((generatedRoot.w_209_4 ctx) i), (((generatedRoot.w_209_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h209_5 i
  ⟨w, parallel_generatedRoot_209.facts _ h⟩

theorem generatedRoot.h213_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_213.Witness, parallel_generatedRoot_213.Facts (⟨((generatedRoot.w_212_0 ctx), ((generatedRoot.w_212_1 ctx), ((generatedRoot.w_212_2 ctx), ((generatedRoot.w_212_3 ctx), ((generatedRoot.w_212_4 ctx), ((generatedRoot.w_212_5 ctx), ())))))), w⟩ : parallel_generatedRoot_213.Ctx backend hashModel tape (path ++ [213, i.val]) params i) (((generatedRoot.w_213_0 ctx) i), (((generatedRoot.w_213_1 ctx) i), (((generatedRoot.w_213_2 ctx) i), (((generatedRoot.w_213_3 ctx) i), (((generatedRoot.w_213_4 ctx) i), (((generatedRoot.w_213_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h213_5 i
  ⟨w, parallel_generatedRoot_213.facts _ h⟩

theorem generatedRoot.h217_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_217.Witness, parallel_generatedRoot_217.Facts (⟨((generatedRoot.w_216_0 ctx), ((generatedRoot.w_216_1 ctx), ((generatedRoot.w_216_2 ctx), ((generatedRoot.w_216_3 ctx), ((generatedRoot.w_216_4 ctx), ((generatedRoot.w_216_5 ctx), ())))))), w⟩ : parallel_generatedRoot_217.Ctx backend hashModel tape (path ++ [217, i.val]) params i) (((generatedRoot.w_217_0 ctx) i), (((generatedRoot.w_217_1 ctx) i), (((generatedRoot.w_217_2 ctx) i), (((generatedRoot.w_217_3 ctx) i), (((generatedRoot.w_217_4 ctx) i), (((generatedRoot.w_217_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h217_5 i
  ⟨w, parallel_generatedRoot_217.facts _ h⟩

theorem generatedRoot.h221_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_221.Witness, parallel_generatedRoot_221.Facts (⟨((generatedRoot.w_220_0 ctx), ((generatedRoot.w_220_1 ctx), ((generatedRoot.w_220_2 ctx), ((generatedRoot.w_220_3 ctx), ((generatedRoot.w_220_4 ctx), ((generatedRoot.w_220_5 ctx), ())))))), w⟩ : parallel_generatedRoot_221.Ctx backend hashModel tape (path ++ [221, i.val]) params i) (((generatedRoot.w_221_0 ctx) i), (((generatedRoot.w_221_1 ctx) i), (((generatedRoot.w_221_2 ctx) i), (((generatedRoot.w_221_3 ctx) i), (((generatedRoot.w_221_4 ctx) i), (((generatedRoot.w_221_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h221_5 i
  ⟨w, parallel_generatedRoot_221.facts _ h⟩

theorem generatedRoot.h225_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_225.Witness, parallel_generatedRoot_225.Facts (⟨((generatedRoot.w_224_0 ctx), ((generatedRoot.w_224_1 ctx), ((generatedRoot.w_224_2 ctx), ((generatedRoot.w_224_3 ctx), ((generatedRoot.w_224_4 ctx), ((generatedRoot.w_224_5 ctx), ())))))), w⟩ : parallel_generatedRoot_225.Ctx backend hashModel tape (path ++ [225, i.val]) params i) (((generatedRoot.w_225_0 ctx) i), (((generatedRoot.w_225_1 ctx) i), (((generatedRoot.w_225_2 ctx) i), (((generatedRoot.w_225_3 ctx) i), (((generatedRoot.w_225_4 ctx) i), (((generatedRoot.w_225_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h225_5 i
  ⟨w, parallel_generatedRoot_225.facts _ h⟩

theorem generatedRoot.h229_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_229.Witness, parallel_generatedRoot_229.Facts (⟨((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_1 ctx), ((generatedRoot.w_228_2 ctx), ((generatedRoot.w_228_3 ctx), ((generatedRoot.w_228_4 ctx), ((generatedRoot.w_228_5 ctx), ())))))), w⟩ : parallel_generatedRoot_229.Ctx backend hashModel tape (path ++ [229, i.val]) params i) (((generatedRoot.w_229_0 ctx) i), (((generatedRoot.w_229_1 ctx) i), (((generatedRoot.w_229_2 ctx) i), (((generatedRoot.w_229_3 ctx) i), (((generatedRoot.w_229_4 ctx) i), (((generatedRoot.w_229_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c7.h229_5 i
  ⟨w, parallel_generatedRoot_229.facts _ h⟩

theorem generatedRoot.h233_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_233.Witness, parallel_generatedRoot_233.Facts (⟨((generatedRoot.w_232_0 ctx), ((generatedRoot.w_232_1 ctx), ((generatedRoot.w_232_2 ctx), ((generatedRoot.w_232_3 ctx), ((generatedRoot.w_232_4 ctx), ((generatedRoot.w_232_5 ctx), ())))))), w⟩ : parallel_generatedRoot_233.Ctx backend hashModel tape (path ++ [233, i.val]) params i) (((generatedRoot.w_233_0 ctx) i), (((generatedRoot.w_233_1 ctx) i), (((generatedRoot.w_233_2 ctx) i), (((generatedRoot.w_233_3 ctx) i), (((generatedRoot.w_233_4 ctx) i), (((generatedRoot.w_233_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h233_5 i
  ⟨w, parallel_generatedRoot_233.facts _ h⟩

theorem generatedRoot.h237_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_237.Witness, parallel_generatedRoot_237.Facts (⟨((generatedRoot.w_236_0 ctx), ((generatedRoot.w_236_1 ctx), ((generatedRoot.w_236_2 ctx), ((generatedRoot.w_236_3 ctx), ((generatedRoot.w_236_4 ctx), ((generatedRoot.w_236_5 ctx), ())))))), w⟩ : parallel_generatedRoot_237.Ctx backend hashModel tape (path ++ [237, i.val]) params i) (((generatedRoot.w_237_0 ctx) i), (((generatedRoot.w_237_1 ctx) i), (((generatedRoot.w_237_2 ctx) i), (((generatedRoot.w_237_3 ctx) i), (((generatedRoot.w_237_4 ctx) i), (((generatedRoot.w_237_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h237_5 i
  ⟨w, parallel_generatedRoot_237.facts _ h⟩

theorem generatedRoot.h241_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_241.Witness, parallel_generatedRoot_241.Facts (⟨((generatedRoot.w_240_0 ctx), ((generatedRoot.w_240_1 ctx), ((generatedRoot.w_240_2 ctx), ((generatedRoot.w_240_3 ctx), ((generatedRoot.w_240_4 ctx), ((generatedRoot.w_240_5 ctx), ())))))), w⟩ : parallel_generatedRoot_241.Ctx backend hashModel tape (path ++ [241, i.val]) params i) (((generatedRoot.w_241_0 ctx) i), (((generatedRoot.w_241_1 ctx) i), (((generatedRoot.w_241_2 ctx) i), (((generatedRoot.w_241_3 ctx) i), (((generatedRoot.w_241_4 ctx) i), (((generatedRoot.w_241_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h241_5 i
  ⟨w, parallel_generatedRoot_241.facts _ h⟩

theorem generatedRoot.h245_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_245.Witness, parallel_generatedRoot_245.Facts (⟨((generatedRoot.w_244_0 ctx), ((generatedRoot.w_244_1 ctx), ((generatedRoot.w_244_2 ctx), ((generatedRoot.w_244_3 ctx), ((generatedRoot.w_244_4 ctx), ((generatedRoot.w_244_5 ctx), ())))))), w⟩ : parallel_generatedRoot_245.Ctx backend hashModel tape (path ++ [245, i.val]) params i) (((generatedRoot.w_245_0 ctx) i), (((generatedRoot.w_245_1 ctx) i), (((generatedRoot.w_245_2 ctx) i), (((generatedRoot.w_245_3 ctx) i), (((generatedRoot.w_245_4 ctx) i), (((generatedRoot.w_245_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h245_5 i
  ⟨w, parallel_generatedRoot_245.facts _ h⟩

theorem generatedRoot.h249_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_249.Witness, parallel_generatedRoot_249.Facts (⟨((generatedRoot.w_248_0 ctx), ((generatedRoot.w_248_1 ctx), ((generatedRoot.w_248_2 ctx), ((generatedRoot.w_248_3 ctx), ((generatedRoot.w_248_4 ctx), ((generatedRoot.w_248_5 ctx), ())))))), w⟩ : parallel_generatedRoot_249.Ctx backend hashModel tape (path ++ [249, i.val]) params i) (((generatedRoot.w_249_0 ctx) i), (((generatedRoot.w_249_1 ctx) i), (((generatedRoot.w_249_2 ctx) i), (((generatedRoot.w_249_3 ctx) i), (((generatedRoot.w_249_4 ctx) i), (((generatedRoot.w_249_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h249_5 i
  ⟨w, parallel_generatedRoot_249.facts _ h⟩

theorem generatedRoot.h253_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_253.Witness, parallel_generatedRoot_253.Facts (⟨((generatedRoot.w_252_0 ctx), ((generatedRoot.w_252_1 ctx), ((generatedRoot.w_252_2 ctx), ((generatedRoot.w_252_3 ctx), ((generatedRoot.w_252_4 ctx), ((generatedRoot.w_252_5 ctx), ())))))), w⟩ : parallel_generatedRoot_253.Ctx backend hashModel tape (path ++ [253, i.val]) params i) (((generatedRoot.w_253_0 ctx) i), (((generatedRoot.w_253_1 ctx) i), (((generatedRoot.w_253_2 ctx) i), (((generatedRoot.w_253_3 ctx) i), (((generatedRoot.w_253_4 ctx) i), (((generatedRoot.w_253_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h253_5 i
  ⟨w, parallel_generatedRoot_253.facts _ h⟩

theorem generatedRoot.h257_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_257.Witness, parallel_generatedRoot_257.Facts (⟨((generatedRoot.w_256_0 ctx), ((generatedRoot.w_256_1 ctx), ((generatedRoot.w_256_2 ctx), ((generatedRoot.w_256_3 ctx), ((generatedRoot.w_256_4 ctx), ((generatedRoot.w_256_5 ctx), ())))))), w⟩ : parallel_generatedRoot_257.Ctx backend hashModel tape (path ++ [257, i.val]) params i) (((generatedRoot.w_257_0 ctx) i), (((generatedRoot.w_257_1 ctx) i), (((generatedRoot.w_257_2 ctx) i), (((generatedRoot.w_257_3 ctx) i), (((generatedRoot.w_257_4 ctx) i), (((generatedRoot.w_257_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h257_5 i
  ⟨w, parallel_generatedRoot_257.facts _ h⟩

theorem generatedRoot.h261_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_261.Witness, parallel_generatedRoot_261.Facts (⟨((generatedRoot.w_260_0 ctx), ((generatedRoot.w_260_1 ctx), ((generatedRoot.w_260_2 ctx), ((generatedRoot.w_260_3 ctx), ((generatedRoot.w_260_4 ctx), ((generatedRoot.w_260_5 ctx), ())))))), w⟩ : parallel_generatedRoot_261.Ctx backend hashModel tape (path ++ [261, i.val]) params i) (((generatedRoot.w_261_0 ctx) i), (((generatedRoot.w_261_1 ctx) i), (((generatedRoot.w_261_2 ctx) i), (((generatedRoot.w_261_3 ctx) i), (((generatedRoot.w_261_4 ctx) i), (((generatedRoot.w_261_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c8.h261_5 i
  ⟨w, parallel_generatedRoot_261.facts _ h⟩

theorem generatedRoot.h265_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_265.Witness, parallel_generatedRoot_265.Facts (⟨((generatedRoot.w_264_0 ctx), ((generatedRoot.w_264_1 ctx), ((generatedRoot.w_264_2 ctx), ((generatedRoot.w_264_3 ctx), ((generatedRoot.w_264_4 ctx), ((generatedRoot.w_264_5 ctx), ())))))), w⟩ : parallel_generatedRoot_265.Ctx backend hashModel tape (path ++ [265, i.val]) params i) (((generatedRoot.w_265_0 ctx) i), (((generatedRoot.w_265_1 ctx) i), (((generatedRoot.w_265_2 ctx) i), (((generatedRoot.w_265_3 ctx) i), (((generatedRoot.w_265_4 ctx) i), (((generatedRoot.w_265_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h265_5 i
  ⟨w, parallel_generatedRoot_265.facts _ h⟩

theorem generatedRoot.h269_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_269.Witness, parallel_generatedRoot_269.Facts (⟨((generatedRoot.w_268_0 ctx), ((generatedRoot.w_268_1 ctx), ((generatedRoot.w_268_2 ctx), ((generatedRoot.w_268_3 ctx), ((generatedRoot.w_268_4 ctx), ((generatedRoot.w_268_5 ctx), ())))))), w⟩ : parallel_generatedRoot_269.Ctx backend hashModel tape (path ++ [269, i.val]) params i) (((generatedRoot.w_269_0 ctx) i), (((generatedRoot.w_269_1 ctx) i), (((generatedRoot.w_269_2 ctx) i), (((generatedRoot.w_269_3 ctx) i), (((generatedRoot.w_269_4 ctx) i), (((generatedRoot.w_269_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h269_5 i
  ⟨w, parallel_generatedRoot_269.facts _ h⟩

theorem generatedRoot.h273_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_273.Witness, parallel_generatedRoot_273.Facts (⟨((generatedRoot.w_272_0 ctx), ((generatedRoot.w_272_1 ctx), ((generatedRoot.w_272_2 ctx), ((generatedRoot.w_272_3 ctx), ((generatedRoot.w_272_4 ctx), ((generatedRoot.w_272_5 ctx), ())))))), w⟩ : parallel_generatedRoot_273.Ctx backend hashModel tape (path ++ [273, i.val]) params i) (((generatedRoot.w_273_0 ctx) i), (((generatedRoot.w_273_1 ctx) i), (((generatedRoot.w_273_2 ctx) i), (((generatedRoot.w_273_3 ctx) i), (((generatedRoot.w_273_4 ctx) i), (((generatedRoot.w_273_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h273_5 i
  ⟨w, parallel_generatedRoot_273.facts _ h⟩

theorem generatedRoot.h277_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_277.Witness, parallel_generatedRoot_277.Facts (⟨((generatedRoot.w_276_0 ctx), ((generatedRoot.w_276_1 ctx), ((generatedRoot.w_276_2 ctx), ((generatedRoot.w_276_3 ctx), ((generatedRoot.w_276_4 ctx), ((generatedRoot.w_276_5 ctx), ())))))), w⟩ : parallel_generatedRoot_277.Ctx backend hashModel tape (path ++ [277, i.val]) params i) (((generatedRoot.w_277_0 ctx) i), (((generatedRoot.w_277_1 ctx) i), (((generatedRoot.w_277_2 ctx) i), (((generatedRoot.w_277_3 ctx) i), (((generatedRoot.w_277_4 ctx) i), (((generatedRoot.w_277_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h277_5 i
  ⟨w, parallel_generatedRoot_277.facts _ h⟩

theorem generatedRoot.h281_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_281.Witness, parallel_generatedRoot_281.Facts (⟨((generatedRoot.w_280_0 ctx), ((generatedRoot.w_280_1 ctx), ((generatedRoot.w_280_2 ctx), ((generatedRoot.w_280_3 ctx), ((generatedRoot.w_280_4 ctx), ((generatedRoot.w_280_5 ctx), ())))))), w⟩ : parallel_generatedRoot_281.Ctx backend hashModel tape (path ++ [281, i.val]) params i) (((generatedRoot.w_281_0 ctx) i), (((generatedRoot.w_281_1 ctx) i), (((generatedRoot.w_281_2 ctx) i), (((generatedRoot.w_281_3 ctx) i), (((generatedRoot.w_281_4 ctx) i), (((generatedRoot.w_281_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h281_5 i
  ⟨w, parallel_generatedRoot_281.facts _ h⟩

theorem generatedRoot.h285_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_285.Witness, parallel_generatedRoot_285.Facts (⟨((generatedRoot.w_284_0 ctx), ((generatedRoot.w_284_1 ctx), ((generatedRoot.w_284_2 ctx), ((generatedRoot.w_284_3 ctx), ((generatedRoot.w_284_4 ctx), ((generatedRoot.w_284_5 ctx), ())))))), w⟩ : parallel_generatedRoot_285.Ctx backend hashModel tape (path ++ [285, i.val]) params i) (((generatedRoot.w_285_0 ctx) i), (((generatedRoot.w_285_1 ctx) i), (((generatedRoot.w_285_2 ctx) i), (((generatedRoot.w_285_3 ctx) i), (((generatedRoot.w_285_4 ctx) i), (((generatedRoot.w_285_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h285_5 i
  ⟨w, parallel_generatedRoot_285.facts _ h⟩

theorem generatedRoot.h289_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_289.Witness, parallel_generatedRoot_289.Facts (⟨((generatedRoot.w_288_0 ctx), ((generatedRoot.w_288_1 ctx), ((generatedRoot.w_288_2 ctx), ((generatedRoot.w_288_3 ctx), ((generatedRoot.w_288_4 ctx), ((generatedRoot.w_288_5 ctx), ())))))), w⟩ : parallel_generatedRoot_289.Ctx backend hashModel tape (path ++ [289, i.val]) params i) (((generatedRoot.w_289_0 ctx) i), (((generatedRoot.w_289_1 ctx) i), (((generatedRoot.w_289_2 ctx) i), (((generatedRoot.w_289_3 ctx) i), (((generatedRoot.w_289_4 ctx) i), (((generatedRoot.w_289_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h289_5 i
  ⟨w, parallel_generatedRoot_289.facts _ h⟩

theorem generatedRoot.h293_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_293.Witness, parallel_generatedRoot_293.Facts (⟨((generatedRoot.w_292_0 ctx), ((generatedRoot.w_292_1 ctx), ((generatedRoot.w_292_2 ctx), ((generatedRoot.w_292_3 ctx), ((generatedRoot.w_292_4 ctx), ((generatedRoot.w_292_5 ctx), ())))))), w⟩ : parallel_generatedRoot_293.Ctx backend hashModel tape (path ++ [293, i.val]) params i) (((generatedRoot.w_293_0 ctx) i), (((generatedRoot.w_293_1 ctx) i), (((generatedRoot.w_293_2 ctx) i), (((generatedRoot.w_293_3 ctx) i), (((generatedRoot.w_293_4 ctx) i), (((generatedRoot.w_293_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c9.h293_5 i
  ⟨w, parallel_generatedRoot_293.facts _ h⟩

theorem generatedRoot.h297_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_297.Witness, parallel_generatedRoot_297.Facts (⟨((generatedRoot.w_296_0 ctx), ((generatedRoot.w_296_1 ctx), ((generatedRoot.w_296_2 ctx), ((generatedRoot.w_296_3 ctx), ((generatedRoot.w_296_4 ctx), ((generatedRoot.w_296_5 ctx), ())))))), w⟩ : parallel_generatedRoot_297.Ctx backend hashModel tape (path ++ [297, i.val]) params i) (((generatedRoot.w_297_0 ctx) i), (((generatedRoot.w_297_1 ctx) i), (((generatedRoot.w_297_2 ctx) i), (((generatedRoot.w_297_3 ctx) i), (((generatedRoot.w_297_4 ctx) i), (((generatedRoot.w_297_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h297_5 i
  ⟨w, parallel_generatedRoot_297.facts _ h⟩

theorem generatedRoot.h301_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_301.Witness, parallel_generatedRoot_301.Facts (⟨((generatedRoot.w_300_0 ctx), ((generatedRoot.w_300_1 ctx), ((generatedRoot.w_300_2 ctx), ((generatedRoot.w_300_3 ctx), ((generatedRoot.w_300_4 ctx), ((generatedRoot.w_300_5 ctx), ())))))), w⟩ : parallel_generatedRoot_301.Ctx backend hashModel tape (path ++ [301, i.val]) params i) (((generatedRoot.w_301_0 ctx) i), (((generatedRoot.w_301_1 ctx) i), (((generatedRoot.w_301_2 ctx) i), (((generatedRoot.w_301_3 ctx) i), (((generatedRoot.w_301_4 ctx) i), (((generatedRoot.w_301_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h301_5 i
  ⟨w, parallel_generatedRoot_301.facts _ h⟩

theorem generatedRoot.h305_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_305.Witness, parallel_generatedRoot_305.Facts (⟨((generatedRoot.w_304_0 ctx), ((generatedRoot.w_304_1 ctx), ((generatedRoot.w_304_2 ctx), ((generatedRoot.w_304_3 ctx), ((generatedRoot.w_304_4 ctx), ((generatedRoot.w_304_5 ctx), ())))))), w⟩ : parallel_generatedRoot_305.Ctx backend hashModel tape (path ++ [305, i.val]) params i) (((generatedRoot.w_305_0 ctx) i), (((generatedRoot.w_305_1 ctx) i), (((generatedRoot.w_305_2 ctx) i), (((generatedRoot.w_305_3 ctx) i), (((generatedRoot.w_305_4 ctx) i), (((generatedRoot.w_305_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h305_5 i
  ⟨w, parallel_generatedRoot_305.facts _ h⟩

theorem generatedRoot.h309_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_309.Witness, parallel_generatedRoot_309.Facts (⟨((generatedRoot.w_308_0 ctx), ((generatedRoot.w_308_1 ctx), ((generatedRoot.w_308_2 ctx), ((generatedRoot.w_308_3 ctx), ((generatedRoot.w_308_4 ctx), ((generatedRoot.w_308_5 ctx), ())))))), w⟩ : parallel_generatedRoot_309.Ctx backend hashModel tape (path ++ [309, i.val]) params i) (((generatedRoot.w_309_0 ctx) i), (((generatedRoot.w_309_1 ctx) i), (((generatedRoot.w_309_2 ctx) i), (((generatedRoot.w_309_3 ctx) i), (((generatedRoot.w_309_4 ctx) i), (((generatedRoot.w_309_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h309_5 i
  ⟨w, parallel_generatedRoot_309.facts _ h⟩

theorem generatedRoot.h310_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_310.Witness, parallel_generatedRoot_310.Facts (⟨((generatedRoot.w_128_0 ctx), ((generatedRoot.w_129_0 ctx), ((generatedRoot.w_130_0 ctx), ((generatedRoot.w_131_0 ctx), ((generatedRoot.w_132_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_133_0 ctx), (((generatedRoot.w_309_4 ctx) i), (((generatedRoot.w_309_0 ctx) i), (((generatedRoot.w_309_2 ctx) i), (((generatedRoot.w_309_5 ctx) i), (((generatedRoot.w_309_1 ctx) i), (((generatedRoot.w_309_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_310.Ctx backend hashModel tape (path ++ [310, i.val]) params i) (((generatedRoot.w_310_0 ctx) i), (((generatedRoot.w_310_1 ctx) i), (((generatedRoot.w_310_2 ctx) i), (((generatedRoot.w_310_3 ctx) i), (((generatedRoot.w_310_4 ctx) i), (((generatedRoot.w_310_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c10.h310_5 i
  ⟨w, parallel_generatedRoot_310.facts _ h⟩

end

end Stage_keys_online_1_1

end
