import Requirement_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Requirement_0

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : (Fin 4 → Int) × (Fin 512 → Int) × (Fin 4 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_1_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → Int :=
  ctx.inputs.1

def generatedRoot.w_5_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 512 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_20_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → Int :=
  ctx.inputs.2.2.1

def generatedRoot.w_0_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  true

def generatedRoot.w_2_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  if (generatedRoot.w_2_0 ctx) then 1 else 0

def generatedRoot.w_4_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  true

def generatedRoot.w_6_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  if (generatedRoot.w_6_0 ctx) then 1 else 0

def generatedRoot.w_8_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_3_0 ctx) + (generatedRoot.w_7_0 ctx))

def generatedRoot.w_9_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  2

def generatedRoot.w_10_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  decide ((generatedRoot.w_8_0 ctx) = (generatedRoot.w_9_0 ctx))

def generatedRoot.w_11_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  if (generatedRoot.w_10_0 ctx) then 1 else 0

def generatedRoot.w_12_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  true

def generatedRoot.w_13_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  if (generatedRoot.w_13_0 ctx) then 1 else 0

def generatedRoot.w_15_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_11_0 ctx) + (generatedRoot.w_14_0 ctx))

def generatedRoot.w_16_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  2

def generatedRoot.w_17_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  decide ((generatedRoot.w_15_0 ctx) = (generatedRoot.w_16_0 ctx))

def generatedRoot.w_18_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  if (generatedRoot.w_17_0 ctx) then 1 else 0

def generatedRoot.w_19_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_21_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  0

def generatedRoot.w_23_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  decide ((generatedRoot.w_21_0 ctx) = (generatedRoot.w_22_0 ctx))

def generatedRoot.w_24_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  if (generatedRoot.w_23_0 ctx) then 1 else 0

def generatedRoot.w_25_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  ((generatedRoot.w_18_0 ctx) + (generatedRoot.w_24_0 ctx))

def generatedRoot.w_26_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Int :=
  2

def generatedRoot.w_27_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  decide ((generatedRoot.w_25_0 ctx) = (generatedRoot.w_26_0 ctx))

def generatedRoot.outs {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Bool :=
  (generatedRoot.w_27_0 ctx)

structure generatedRoot.Facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Bool) : Prop where
  t0 : 0 ≤ (4)
  h2_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_2 tape (path ++ [2, i]) params i (current, ((generatedRoot.w_1_0 ctx), ())) next) (Int.toNat (4)) (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx)
  t2 : 0 ≤ (384)
  h6_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 tape (path ++ [6, i]) params i (current, ((generatedRoot.w_5_0 ctx), ())) next) (Int.toNat (384)) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx)
  t4 : 0 ≤ (128)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_13 tape (path ++ [13, i]) params i (current, ((generatedRoot.w_5_0 ctx), ())) next) (Int.toNat (128)) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx)
  t6 : 0 ≤ (4)
  h21_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_21 tape (path ++ [21, i]) params i (current, ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_20_0 ctx), ())))) next) (Int.toNat (4)) (generatedRoot.w_19_0 ctx) (generatedRoot.w_21_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : Bool}
    (h : generatedRoot.constraints_0 tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_27_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : Bool) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) {outputs : Bool}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Requirement_0

end
