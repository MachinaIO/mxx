import W_Stage_payload

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace W_Stage_payload

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_0_0

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (generatedRoot.w_0_0 ctx)

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) : Prop where
  h0_0 : MxxRuntime.uniformIntervalSample (-1) (1) (generatedRoot.w_0_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1}
    (h : generatedRoot.constraints_0 (generatedRoot.w_0_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

end W_Stage_payload

end
