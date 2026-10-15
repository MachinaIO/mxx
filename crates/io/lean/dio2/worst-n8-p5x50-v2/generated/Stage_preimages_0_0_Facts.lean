import Stage_preimages_0_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_preimages_0_0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 54 25

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (params : Params) where
  inputs : MxxRuntime.TrapdoorValue (mxxMat0) Unit × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × Unit
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.inputs.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat1 :=
  ctx.inputs.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat1 :=
  ctx.inputs.2.2.1

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.trapdoorPublic (generatedRoot.w_0_0 ctx)

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat2 :=
  ctx.witness.w_3_0

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat2 :=
  ctx.witness.w_5_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat2 :=
  ctx.witness.w_7_0

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat2 :=
  ctx.witness.w_9_0

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat2 :=
  ctx.witness.w_11_0

def generatedRoot.outs  (ctx : generatedRootCtx) : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit :=
  ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_11_0 ctx), ())))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit) : Prop where
  t0 : (36) = 36
  h3_0 : (∀ i : Fin 36, parallel_generatedRoot_3 backend params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_2_0 ctx) i), ()))) ((generatedRoot.w_3_0 ctx) i))
  t2 : (36) = 36
  h5_0 : (∀ i : Fin 36, parallel_generatedRoot_5 backend params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_4_0 ctx) i), ()))) ((generatedRoot.w_5_0 ctx) i))
  t4 : (36) = 36
  h7_0 : (∀ i : Fin 36, parallel_generatedRoot_7 backend params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_6_0 ctx) i), ()))) ((generatedRoot.w_7_0 ctx) i))
  t6 : (36) = 36
  h9_0 : (∀ i : Fin 36, parallel_generatedRoot_9 backend params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_8_0 ctx) i), ()))) ((generatedRoot.w_9_0 ctx) i))
  t8 : (36) = 36
  h11_0 : (∀ i : Fin 36, parallel_generatedRoot_11 backend params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_10_0 ctx) i), ()))) ((generatedRoot.w_11_0 ctx) i))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit}
    (h : generatedRoot.constraints_0 backend params (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit}
    (h : generatedRoot.body backend params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

end Stage_preimages_0_0

end
