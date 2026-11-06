import Stage_setup_trapdoor

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_setup_trapdoor

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 295

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : Unit
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend tape path params)

def generatedRoot.w_0_1  (ctx : generatedRootCtx) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.witness.w_0_1

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_0_0

def generatedRoot.outs  (ctx : generatedRootCtx) : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_1 ctx), ()))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop where
  h0_1 : MxxRuntime.statTrapdoorSampleAt backend tape (path ++ [0]) 265 (5154369773525533/1125899906842624) 4096 30 2610114297 (generatedRoot.w_0_0 ctx) (generatedRoot.w_0_1 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit}
    (h : generatedRoot.constraints_0 backend tape path (generatedRoot.w_0_1 ctx) (generatedRoot.w_0_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit}
    (h : generatedRoot.body backend tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

end Stage_setup_trapdoor

end
