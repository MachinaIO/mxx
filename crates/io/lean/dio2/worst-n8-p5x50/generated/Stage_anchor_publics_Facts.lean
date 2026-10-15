import Stage_anchor_publics

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_anchor_publics

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : MxxRuntime.TrapdoorValue (mxxMat0) Unit × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.inputs.1

def generatedRoot.w_2_0 {params : Params} (ctx : generatedRoot.Ctx params) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.inputs.2.1

def generatedRoot.w_1_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  MxxRuntime.trapdoorPublic (generatedRoot.w_0_0 ctx)

def generatedRoot.w_3_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  MxxRuntime.trapdoorPublic (generatedRoot.w_2_0 ctx)

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 × mxxMat0 × Unit :=
  ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_3_0 ctx), ()))

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop where
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : mxxMat0 × mxxMat0 × Unit}
    (h : generatedRoot.constraints_0 (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : mxxMat0 × mxxMat0 × Unit}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Stage_anchor_publics

end
