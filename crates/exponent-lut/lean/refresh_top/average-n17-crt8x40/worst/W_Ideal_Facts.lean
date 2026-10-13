import W_Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Ideal

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : Int
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.inputs

def generatedRoot.w_1_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  0

def generatedRoot.w_2_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_0_0 ctx) * (generatedRoot.w_1_0 ctx))

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  (generatedRoot.w_2_0 ctx)

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Int) : Prop where
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Int}
    (h : generatedRoot.constraints_0 (generatedRoot.w_2_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : Int}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end W_Ideal

end
