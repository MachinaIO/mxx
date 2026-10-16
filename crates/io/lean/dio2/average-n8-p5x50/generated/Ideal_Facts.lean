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
  inputs : Int × (Fin 2 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ctx.inputs.1

def generatedRoot.w_3_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_1_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_2_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_0_0 ctx) + (generatedRoot.w_1_0 ctx))

def generatedRoot.w_4_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ctx.witness.w_4_0

def generatedRoot.w_5_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  1

def generatedRoot.w_6_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_4_0 ctx) * (generatedRoot.w_5_0 ctx))

def generatedRoot.w_7_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_2_0 ctx) + (generatedRoot.w_6_0 ctx))

def generatedRoot.w_8_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  1

def generatedRoot.w_9_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_10_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_8_0 ctx) + (generatedRoot.w_9_0 ctx))

def generatedRoot.w_11_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_12_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_13_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_11_0 ctx) + (generatedRoot.w_12_0 ctx))

def generatedRoot.w_14_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  1

def generatedRoot.w_15_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_16_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_14_0 ctx) + (generatedRoot.w_15_0 ctx))

def generatedRoot.w_17_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_18_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_19_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_17_0 ctx) + (generatedRoot.w_18_0 ctx))

def generatedRoot.w_20_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  1

def generatedRoot.w_22_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_20_0 ctx) + (generatedRoot.w_21_0 ctx))

def generatedRoot.w_23_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ctx.witness.w_23_0

def generatedRoot.w_24_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  1

def generatedRoot.w_25_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_23_0 ctx) * (generatedRoot.w_24_0 ctx))

def generatedRoot.w_26_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_22_0 ctx) + (generatedRoot.w_25_0 ctx))

def generatedRoot.w_27_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_28_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_29_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_27_0 ctx) + (generatedRoot.w_28_0 ctx))

def generatedRoot.w_30_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  1

def generatedRoot.w_31_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_32_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_30_0 ctx) + (generatedRoot.w_31_0 ctx))

def generatedRoot.w_33_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  1

def generatedRoot.w_34_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_35_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_33_0 ctx) + (generatedRoot.w_34_0 ctx))

def generatedRoot.w_36_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_37_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_38_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_36_0 ctx) + (generatedRoot.w_37_0 ctx))

def generatedRoot.w_39_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ctx.witness.w_39_0

def generatedRoot.outs {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  (generatedRoot.w_39_0 ctx)

structure generatedRoot.Facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Int) : Prop where
  h4_0 : MxxRuntime.familyGetStatic (generatedRoot.w_3_0 ctx) (0) (generatedRoot.w_4_0 ctx)
  h7_0 : 0 ≤ (generatedRoot.w_7_0 ctx)
  h7_0_1 : (generatedRoot.w_7_0 ctx) < 4
  t3 : 4 = 4
  h20_0 : MxxRuntime.select (generatedRoot.w_7_0 ctx) [(generatedRoot.w_10_0 ctx), (generatedRoot.w_13_0 ctx), (generatedRoot.w_16_0 ctx), (generatedRoot.w_19_0 ctx)] (generatedRoot.w_20_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (generatedRoot.w_3_0 ctx) (1) (generatedRoot.w_23_0 ctx)
  h26_0 : 0 ≤ (generatedRoot.w_26_0 ctx)
  h26_0_1 : (generatedRoot.w_26_0 ctx) < 4
  t8 : 4 = 4
  h39_0 : MxxRuntime.select (generatedRoot.w_26_0 ctx) [(generatedRoot.w_29_0 ctx), (generatedRoot.w_32_0 ctx), (generatedRoot.w_35_0 ctx), (generatedRoot.w_38_0 ctx)] (generatedRoot.w_39_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : Int}
    (h : generatedRoot.constraints_0 (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) {outputs : Int}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Ideal

end
