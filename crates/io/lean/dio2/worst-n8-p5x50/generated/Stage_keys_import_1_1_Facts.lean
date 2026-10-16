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
structure generatedRoot.Ctx (params : Params) where
  inputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → Int) × (Fin 2 → Int) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_2_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.1

def generatedRoot.w_3_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_4_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_5_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_7_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_8_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_9_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_10_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_12_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_13_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_6_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def generatedRoot.w_6_2 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → mxxMat1 :=
  ctx.witness.w_6_2

def generatedRoot.w_11_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_11_0

def generatedRoot.w_11_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_11_1

def generatedRoot.w_14_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_14_0

def generatedRoot.w_14_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_14_1

def generatedRoot.w_15_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_15_0

def generatedRoot.w_15_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_15_1

def generatedRoot.w_16_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_16_0

def generatedRoot.w_16_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_16_1

def generatedRoot.w_17_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_17_0

def generatedRoot.w_17_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_17_1

def generatedRoot.w_18_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_18_0

def generatedRoot.w_18_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_18_1

def generatedRoot.w_19_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_19_0

def generatedRoot.w_19_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_19_1

def generatedRoot.w_20_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_20_0

def generatedRoot.w_20_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_20_1

def generatedRoot.w_21_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_21_0

def generatedRoot.w_21_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_21_1

def generatedRoot.w_22_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_22_0

def generatedRoot.w_22_1 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_22_1

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_15_1 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_16_1 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_17_1 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_18_1 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_19_1 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_20_1 ctx), ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_21_1 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_22_1 ctx), ())))))))))))))))))))))))))))

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop where
  t0 : (2) = 2
  h6_2 : (∀ i : Fin 2, parallel_generatedRoot_6 params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_5_0 ctx) i), ())))) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), ()))))
  t2 : (8) = 8
  h11_1 : (∀ i : Fin 8, parallel_generatedRoot_11 params i (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_10_0 ctx) i), ())) (((generatedRoot.w_11_0 ctx) i), (((generatedRoot.w_11_1 ctx) i), ())))
  t4 : (4) = 4
  h14_1 : (∀ i : Fin 4, parallel_generatedRoot_14 params i (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_13_0 ctx) i), ())) (((generatedRoot.w_14_0 ctx) i), (((generatedRoot.w_14_1 ctx) i), ())))
  t6 : (8) = 8
  h15_1 : (∀ i : Fin 8, parallel_generatedRoot_15 params i (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_10_0 ctx) i), ())) (((generatedRoot.w_15_0 ctx) i), (((generatedRoot.w_15_1 ctx) i), ())))
  t8 : (4) = 4
  h16_1 : (∀ i : Fin 4, parallel_generatedRoot_16 params i (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_13_0 ctx) i), ())) (((generatedRoot.w_16_0 ctx) i), (((generatedRoot.w_16_1 ctx) i), ())))
  t10 : (8) = 8
  h17_1 : (∀ i : Fin 8, parallel_generatedRoot_17 params i (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_10_0 ctx) i), ())) (((generatedRoot.w_17_0 ctx) i), (((generatedRoot.w_17_1 ctx) i), ())))
  t12 : (4) = 4
  h18_1 : (∀ i : Fin 4, parallel_generatedRoot_18 params i (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_13_0 ctx) i), ())) (((generatedRoot.w_18_0 ctx) i), (((generatedRoot.w_18_1 ctx) i), ())))
  t14 : (8) = 8
  h19_1 : (∀ i : Fin 8, parallel_generatedRoot_19 params i (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_10_0 ctx) i), ())) (((generatedRoot.w_19_0 ctx) i), (((generatedRoot.w_19_1 ctx) i), ())))
  t16 : (4) = 4
  h20_1 : (∀ i : Fin 4, parallel_generatedRoot_20 params i (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_13_0 ctx) i), ())) (((generatedRoot.w_20_0 ctx) i), (((generatedRoot.w_20_1 ctx) i), ())))
  t18 : (8) = 8
  h21_1 : (∀ i : Fin 8, parallel_generatedRoot_21 params i (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_10_0 ctx) i), ())) (((generatedRoot.w_21_0 ctx) i), (((generatedRoot.w_21_1 ctx) i), ())))
  t20 : (4) = 4
  h22_1 : (∀ i : Fin 4, parallel_generatedRoot_22 params i (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_13_0 ctx) i), ())) (((generatedRoot.w_22_0 ctx) i), (((generatedRoot.w_22_1 ctx) i), ())))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit}
    (h : generatedRoot.constraints_0 params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_16_1 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_17_1 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_18_1 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_19_1 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_20_1 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_21_1 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_22_1 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Stage_keys_import_1_1

end
