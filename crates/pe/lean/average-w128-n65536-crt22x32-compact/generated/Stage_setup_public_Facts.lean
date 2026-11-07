import Stage_setup_public

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_setup_public

local notation "mxxNat0" => 67598865718973452911259282735194974463966008168342023559927392187609292957345997883634735633837025843140326117569327049993577673305918340947654355600136203031252562778990577685529163020701990351591297285624954881
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 501
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_1 65536 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_1 65536 1 501

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0
  witness : generatedRoot.Witness

def generatedRoot.w_1_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat0 :=
  ctx.inputs

def generatedRoot.w_0_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat1)

def generatedRoot.w_2_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx)

def generatedRoot.w_4_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  ctx.witness.w_4_0

def generatedRoot.w_5_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat0 :=
  ctx.witness.w_5_0

def generatedRoot.w_6_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_6_0 ctx)

def generatedRoot.w_8_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  MxxRuntime.matrixSub (generatedRoot.w_4_0 ctx) (generatedRoot.w_7_0 ctx)

def generatedRoot.w_9_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  MxxRuntime.matrixAdd (generatedRoot.w_3_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.outs {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat2 :=
  (generatedRoot.w_9_0 ctx)

structure generatedRoot.Facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : mxxMat2) : Prop where
  h2_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx)
  h4_0 : MxxRuntime.uniformResidueSampleAt tape (path ++ [4]) (generatedRoot.w_4_0 ctx)
  h5_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx)
  h6_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : mxxMat2}
    (h : generatedRoot.constraints_0 tape path (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_9_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩

structure generatedRoot.Facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : mxxMat2) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) {outputs : mxxMat2}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Stage_setup_public

end
