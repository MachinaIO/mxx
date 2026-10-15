import W_Stage_decode

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Stage_decode

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : mxxMat0
  witness : generatedRoot.Witness

def generatedRoot.w_1_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  ctx.inputs

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  0

def generatedRoot.w_2_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → Int :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_3_0

def generatedRoot.w_4_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_5_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_3_0 ctx) / (generatedRoot.w_4_0 ctx))

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  (generatedRoot.w_5_0 ctx)

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Int) : Prop where
  h2_0 : MxxRuntime.polynomialValues false (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx)
  t1 : 0 ≤ (8)
  h3_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_3 params i (current, ((generatedRoot.w_2_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_0_0 ctx) (generatedRoot.w_3_0 ctx)
  h4_0 : (generatedRoot.w_4_0 ctx) ≠ 0
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Int}
    (h : generatedRoot.constraints_0 params (generatedRoot.w_1_0 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : Int}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end W_Stage_decode

end
