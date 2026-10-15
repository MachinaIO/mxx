import W_Stage_anchors

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Stage_anchors

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (params : Params) where
  inputs : Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_1 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.witness.w_0_1

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_0_0

def generatedRoot.w_1_1 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.witness.w_1_1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_1_0

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_1 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_1_1 ctx), ()))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop where
  h0_1 : MxxRuntime.trapdoorSample backend (4/1) 4096 25 22011706 (generatedRoot.w_0_0 ctx) (generatedRoot.w_0_1 ctx)
  h1_1 : MxxRuntime.trapdoorSample backend (4/1) 4096 25 22011706 (generatedRoot.w_1_0 ctx) (generatedRoot.w_1_1 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {params : Params} {ctx : generatedRoot.Ctx backend params} {outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit}
    (h : generatedRoot.constraints_0 backend (generatedRoot.w_0_1 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_1 ctx) (generatedRoot.w_1_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) {outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit}
    (h : generatedRoot.body backend params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end W_Stage_anchors

end
