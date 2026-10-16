import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Ideal

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : Fin 32768 → Int
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 32768 → Int :=
  ctx.inputs

def generatedRoot.outs {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 32768 → Int :=
  (generatedRoot.w_0_0 ctx)

structure generatedRoot.Facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Fin 32768 → Int) : Prop where
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : Fin 32768 → Int}
    (h : generatedRoot.constraints_0 (generatedRoot.w_0_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h⟩

structure generatedRoot.Facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Fin 32768 → Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) {outputs : Fin 32768 → Int}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Ideal

end
