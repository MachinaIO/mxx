import Stage_keygen_helpers

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_keygen_helpers

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 45

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : Unit
  witness : generatedRoot.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat0 :=
  ctx.witness.w_0_0

def generatedRoot.outs  (ctx : generatedRootCtx) : Fin 256 → mxxMat0 :=
  (generatedRoot.w_0_0 ctx)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : Fin 256 → mxxMat0) : Prop where
  t0 : (256) = 256
  h0_0 : (∀ i : Fin 256, parallel_generatedRoot_0 tape (path ++ [0, i.val]) params i () ((generatedRoot.w_0_0 ctx) i))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : Fin 256 → mxxMat0}
    (h : generatedRoot.constraints_0 tape path params (generatedRoot.w_0_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : Fin 256 → mxxMat0) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : Fin 256 → mxxMat0}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

end Stage_keygen_helpers

end
