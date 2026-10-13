import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace Ideal

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : Fin 65536 → Int
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 65536 → Int :=
  ctx.inputs

def generatedRoot.outs {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 65536 → Int :=
  (generatedRoot.w_0_0 ctx)

structure generatedRoot.Facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Fin 65536 → Int) : Prop where
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : Fin 65536 → Int}
    (h : generatedRoot.constraints_0 (generatedRoot.w_0_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h⟩

structure generatedRoot.Facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Fin 65536 → Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) {outputs : Fin 65536 → Int}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

end Ideal

end
