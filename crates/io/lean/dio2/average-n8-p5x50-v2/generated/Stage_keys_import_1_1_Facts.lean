import Stage_keys_import_1_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_keys_import_1_1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.1

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def generatedRoot.w_6_2  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_6_2

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_11_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_14_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_15_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_16_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_17_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_18_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_19_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_20_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_21_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_22_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_15_1 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_16_1 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_17_1 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_18_1 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_19_1 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_20_1 ctx), ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_21_1 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_22_1 ctx), ())))))))))))))))))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t0 : (2) = 2
  h6_2 : (∀ i : Fin 2, parallel_generatedRoot_6 tape (path ++ [6, i.val]) params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_5_0 ctx) i), ())))) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), ()))))
  t2 : (8) = 8
  t3 : (4) = 4
  t4 : (8) = 8
  t5 : (4) = 4
  t6 : (8) = 8
  t7 : (4) = 4
  t8 : (8) = 8
  t9 : (4) = 4
  t10 : (8) = 8
  t11 : (4) = 4
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_16_1 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_17_1 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_18_1 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_19_1 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_20_1 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_21_1 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_22_1 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

end Stage_keys_import_1_1

end
