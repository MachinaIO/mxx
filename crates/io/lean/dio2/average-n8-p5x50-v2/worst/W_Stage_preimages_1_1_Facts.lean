import W_Stage_preimages_1_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Stage_preimages_1_1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 2 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 54 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (params : Params) where
  inputs : MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.inputs.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.inputs.2.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.trapdoorPublic (generatedRoot.w_0_0 ctx)

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_3_0

def generatedRoot.outs  (ctx : generatedRootCtx) : mxxMat2 :=
  (generatedRoot.w_3_0 ctx)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : mxxMat2) : Prop where
  t0 : 0 ≤ (22011706)
  h3_0 : MxxRuntime.preimageRunsDispatched backend (generatedRoot.w_1_0 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (Int.toNat (22011706)) (generatedRoot.w_3_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : mxxMat2}
    (h : generatedRoot.constraints_0 backend (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : mxxMat2) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : mxxMat2}
    (h : generatedRoot.body backend params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

end W_Stage_preimages_1_1

end
