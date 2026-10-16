import Stage_zero

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_zero

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : Unit
  witness : generatedRoot.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_1_0

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_3_0

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_4_0

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_5_0

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_0

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_8_0

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_9_0

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_0

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_0

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_0

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_15_0

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_17_0

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_18_0

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_19_0

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_24_0 ctx), ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_26_0 ctx), ())))))))))))))))))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  t0 : (1) = 1
  h1_0 : (∀ i : Fin 1, parallel_generatedRoot_1 tape (path ++ [1, i.val]) params i () ((generatedRoot.w_1_0 ctx) i))
  t2 : (1) = 1
  h2_0 : (∀ i : Fin 1, parallel_generatedRoot_2 tape (path ++ [2, i.val]) params i () ((generatedRoot.w_2_0 ctx) i))
  t4 : (1) = 1
  h3_0 : (∀ i : Fin 1, parallel_generatedRoot_3 tape (path ++ [3, i.val]) params i () ((generatedRoot.w_3_0 ctx) i))
  t6 : (1) = 1
  h4_0 : (∀ i : Fin 1, parallel_generatedRoot_4 tape (path ++ [4, i.val]) params i () ((generatedRoot.w_4_0 ctx) i))
  t8 : (1) = 1
  h5_0 : (∀ i : Fin 1, parallel_generatedRoot_5 tape (path ++ [5, i.val]) params i () ((generatedRoot.w_5_0 ctx) i))
  t10 : (1) = 1
  h6_0 : (∀ i : Fin 1, parallel_generatedRoot_6 tape (path ++ [6, i.val]) params i () ((generatedRoot.w_6_0 ctx) i))
  t12 : (1) = 1
  h7_0 : (∀ i : Fin 1, parallel_generatedRoot_7 tape (path ++ [7, i.val]) params i () ((generatedRoot.w_7_0 ctx) i))
  t14 : (1) = 1
  h8_0 : (∀ i : Fin 1, parallel_generatedRoot_8 tape (path ++ [8, i.val]) params i () ((generatedRoot.w_8_0 ctx) i))
  t16 : (1) = 1
  h9_0 : (∀ i : Fin 1, parallel_generatedRoot_9 tape (path ++ [9, i.val]) params i () ((generatedRoot.w_9_0 ctx) i))
  t18 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_generatedRoot_10 tape (path ++ [10, i.val]) params i () ((generatedRoot.w_10_0 ctx) i))
  t20 : (1) = 1
  h11_0 : (∀ i : Fin 1, parallel_generatedRoot_11 tape (path ++ [11, i.val]) params i () ((generatedRoot.w_11_0 ctx) i))
  t22 : (1) = 1
  h12_0 : (∀ i : Fin 1, parallel_generatedRoot_12 tape (path ++ [12, i.val]) params i () ((generatedRoot.w_12_0 ctx) i))
  t24 : (1) = 1
  h13_0 : (∀ i : Fin 1, parallel_generatedRoot_13 tape (path ++ [13, i.val]) params i () ((generatedRoot.w_13_0 ctx) i))
  t26 : (1) = 1
  h14_0 : (∀ i : Fin 1, parallel_generatedRoot_14 tape (path ++ [14, i.val]) params i () ((generatedRoot.w_14_0 ctx) i))
  t28 : (8) = 8
  h15_0 : (∀ i : Fin 8, parallel_generatedRoot_15 tape (path ++ [15, i.val]) params i () ((generatedRoot.w_15_0 ctx) i))
  t30 : (8) = 8
  h16_0 : (∀ i : Fin 8, parallel_generatedRoot_16 tape (path ++ [16, i.val]) params i () ((generatedRoot.w_16_0 ctx) i))

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t32 : (4) = 4
  h17_0 : (∀ i : Fin 4, parallel_generatedRoot_17 tape (path ++ [17, i.val]) params i () ((generatedRoot.w_17_0 ctx) i))
  t34 : (4) = 4
  h18_0 : (∀ i : Fin 4, parallel_generatedRoot_18 tape (path ++ [18, i.val]) params i () ((generatedRoot.w_18_0 ctx) i))
  t36 : (2) = 2
  h19_0 : (∀ i : Fin 2, parallel_generatedRoot_19 tape (path ++ [19, i.val]) params i () ((generatedRoot.w_19_0 ctx) i))
  t38 : (2) = 2
  h20_0 : (∀ i : Fin 2, parallel_generatedRoot_20 tape (path ++ [20, i.val]) params i () ((generatedRoot.w_20_0 ctx) i))
  t40 : (4) = 4
  h21_0 : (∀ i : Fin 4, parallel_generatedRoot_21 tape (path ++ [21, i.val]) params i () ((generatedRoot.w_21_0 ctx) i))
  t42 : (4) = 4
  h22_0 : (∀ i : Fin 4, parallel_generatedRoot_22 tape (path ++ [22, i.val]) params i () ((generatedRoot.w_22_0 ctx) i))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

end Stage_zero

end
