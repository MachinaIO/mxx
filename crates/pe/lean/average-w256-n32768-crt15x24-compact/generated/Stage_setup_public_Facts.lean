import Stage_setup_public

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_setup_public

local notation "mxxNat0" => 1009041927918404748912066953523025233623263629931203392403660676397045184266783237099217072052453834753
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 295
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 295

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0
  witness : generatedRoot.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx tape path params)

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat1)

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx)

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_4_0

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_5_0

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_6_0 ctx)

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixSub (generatedRoot.w_4_0 ctx) (generatedRoot.w_7_0 ctx)

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixAdd (generatedRoot.w_3_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.outs  (ctx : generatedRootCtx) : mxxMat2 :=
  (generatedRoot.w_9_0 ctx)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : mxxMat2) : Prop where
  h2_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx)
  h4_0 : MxxRuntime.uniformResidueSampleAt tape (path ++ [4]) (generatedRoot.w_4_0 ctx)
  h5_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx)
  h6_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : mxxMat2}
    (h : generatedRoot.constraints_0 tape path (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_9_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : mxxMat2) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : mxxMat2}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

end Stage_setup_public

end
