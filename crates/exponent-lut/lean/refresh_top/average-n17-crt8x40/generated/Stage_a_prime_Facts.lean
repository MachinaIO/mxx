import Stage_a_prime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_a_prime

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 28
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 56

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : ByteArray
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) : ByteArray :=
  ctx.inputs

def generatedRoot.w_1_0 {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_1_0

def generatedRoot.w_2_0 {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0 {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_3_0

def generatedRoot.outs {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) : mxxMat1 :=
  (generatedRoot.w_3_0 ctx)

structure generatedRoot.Facts_0 {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) (outputs : mxxMat1) : Prop where
  h1_0 : MxxRuntime.hashSample (hashModel) ([116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 0]) ([]) ((generatedRoot.w_0_0 ctx)) (generatedRoot.w_1_0 ctx)
  h2_0 : MxxRuntime.hashSample (hashModel) ([116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 1]) ([]) ((generatedRoot.w_0_0 ctx)) (generatedRoot.w_2_0 ctx)
  h3_0 : MxxRuntime.concatColumns (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx hashModel tape path params} {outputs : mxxMat1}
    (h : generatedRoot.constraints_0 hashModel (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩

structure generatedRoot.Facts {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) (outputs : mxxMat1) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx hashModel tape path params) {outputs : mxxMat1}
    (h : generatedRoot.body hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Stage_a_prime

end
