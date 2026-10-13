import Stage_setup_trapdoor

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace Stage_setup_trapdoor

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 501

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_1 {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend tape path params) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.witness.w_0_1

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend tape path params) : mxxMat0 :=
  ctx.witness.w_0_0

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend tape path params) : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_0_1 ctx), ()))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend tape path params) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop where
  h0_1 : MxxRuntime.statTrapdoorSampleAt backend tape (path ++ [0]) 457 (5154369773525533/1125899906842624) 65536 44 73534451498 (generatedRoot.w_0_0 ctx) (generatedRoot.w_0_1 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend tape path params} {outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit}
    (h : generatedRoot.constraints_0 backend tape path (generatedRoot.w_0_1 ctx) (generatedRoot.w_0_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend tape path params) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend tape path params) {outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit}
    (h : generatedRoot.body backend tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

end Stage_setup_trapdoor

end
